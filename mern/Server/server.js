// server.js
const express       = require("express");
const mongoose      = require("mongoose");
const cors          = require("cors");
const helmet        = require("helmet");
const compression   = require("compression");
const morgan        = require("morgan");
const multer        = require("multer");
let sharp;
try {
  sharp = require("sharp");
} catch (e) {
  console.warn("Sharp module not found; image resizing disabled.");
  sharp = null;
}
const axios         = require("axios");
const jwt           = require("jsonwebtoken");
const bcrypt        = require("bcrypt");
const path          = require("path");
const fs            = require("fs");

// Config
const CLIENT_URL    = "https://mern-test-client.onrender.com";
const MONGODB_URI   = "mongodb+srv://myAppUser:890iopjklnm@plantdiseasedetection.uhd0o.mongodb.net/?retryWrites=true&w=majority&appName=Plantdiseasedetection";
const FLASK_URL     = "https://predict-app-mawg.onrender.com";
const GEMINI_URL    = "https://agent-app.onrender.com";
const JWT_SECRET    = "your_jwt_secret_here";
const JWT_EXPIRES_IN= "1h";
const PORT          = process.env.PORT || 5000;
const UPLOAD_DIR    = path.join(__dirname, "uploads");

// Ensure uploads folder
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

// Connect MongoDB
mongoose.connect(MONGODB_URI)
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => {
    console.error("❌ MongoDB error:", err);
    process.exit(1);
  });

const app = express();

// Debug: log incoming headers
app.use((req, res, next) => {
  console.log("Incoming Headers:", req.headers);
  next();
});

app.use(helmet());
app.use(compression());
app.use(morgan("combined"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
  origin: CLIENT_URL,
  methods: ["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
  credentials: true,
  allowedHeaders: ["Content-Type","Authorization"]
}));
app.options("*", cors());

// Multer (disk storage)
const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    const id = `${Date.now()}-${Math.round(Math.random()*1e9)}`;
    cb(null, id + path.extname(file.originalname));
  }
});
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    cb(file.mimetype.startsWith("image/") ? null : new Error("Only images"), true);
  }
});

// JWT middleware
const authenticate = (req, res, next) => {
  const header = req.headers.authorization;
  if (!header) return res.status(401).json({ message: "No token provided" });
  const token = header.startsWith("Bearer ") ? header.split(" ")[1] : header;
  if (!token) return res.status(401).json({ message: "Invalid token format" });
  jwt.verify(token, JWT_SECRET, (err, decoded) => {
    if (err) {
      return res.status(401).json({ message: err.name === "TokenExpiredError" ? "Token expired" : "Invalid token" });
    }
    req.userId = decoded.userId;
    next();
  });
};

// User schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true },
  email:    { type: String, unique: true, required: true },
  password: { type: String, required: true },
  history: [{
    plantType:     String,
    status:        String,
    recommendation:String,
    imageUrl:      String,
    thumbnailUrl:  String,
    analyzedAt:    { type: Date, default: Date.now }
  }]
}, { timestamps: true });
const User = mongoose.model("User", userSchema);

// Routes
app.get("/", (_, res) => res.json({ status: "ok" }));

app.post("/register", async (req, res, next) => {
  try {
    const { username, email, password } = req.body;
    if (![username, email, password].every(Boolean))
      return res.status(400).json({ message: "Missing fields" });
    if (await User.exists({ email }))
      return res.status(409).json({ message: "Email in use" });
    const hash = await bcrypt.hash(password, 12);
    await new User({ username, email, password: hash }).save();
    res.status(201).json({ message: "Registered" });
  } catch (e) {
    next(e);
  }
});

app.post("/login", async (req, res, next) => {
  try {
    const { email, password } = req.body;
    if (![email, password].every(Boolean))
      return res.status(400).json({ message: "Missing creds" });
    const user = await User.findOne({ email });
    if (!user || !(await bcrypt.compare(password, user.password)))
      return res.status(401).json({ message: "Invalid creds" });
    const token = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
    res.json({ token });
  } catch (e) {
    next(e);
  }
});

app.post("/analyze", authenticate, upload.single("image"), async (req, res, next) => {
  try {
    const { plantType, waterFreq, language } = req.body;
    if (![plantType, waterFreq].every(Boolean) || !req.file)
      return res.status(400).json({ message: "Missing data or image" });

    const filePath = path.join(UPLOAD_DIR, req.file.filename);
    let buf = fs.readFileSync(filePath);

    if (sharp) {
      buf = await sharp(buf)
        .resize(800, 800, { fit: "inside" })
        .jpeg({ quality: 80 })
        .toBuffer();
      fs.writeFileSync(filePath, buf);
    }

    let thumb = null;
    if (sharp) {
      thumb = `thumb-${req.file.filename}`;
      await sharp(buf)
        .resize(200, 200, { fit: "cover" })
        .toFile(path.join(UPLOAD_DIR, thumb));
    }

    // ——— Prediction call with timeout + error handling ———
    let status;
    try {
      const predictResp = await axios.post(
        `${FLASK_URL}/predict`,
        buf,
        {
          headers: { "Content-Type": "application/octet-stream" },
          timeout: 20_000  // 20s timeout
        }
      );
      const data = predictResp.data;
      status = typeof data === "string"
        ? (data.match(/Prediction:\s*(\w+)/)?.[1] || "Unknown")
        : (data.prediction || data.status || "Unknown");
    } catch (err) {
      console.error("❌ Predict API error:", err.message);
      return res.status(502).json({ message: "Prediction service unavailable" });
    }

    // Recommendation (still best-effort)
    let recommendation = "Unavailable";
    try {
      const rec = await axios.post(
        `${GEMINI_URL}/recommend`,
        { status, plantType, waterFreq: +waterFreq, language },
        { timeout: 20_000 }
      );
      recommendation = rec.data.recommendation;
    } catch (e) {
      console.warn("⚠️ Recommendation API failed:", e.message);
    }

    const imageUrl     = `/uploads/${req.file.filename}`;
    const thumbnailUrl = thumb ? `/uploads/${thumb}` : imageUrl;

    await User.findByIdAndUpdate(req.userId, {
      $push: { history: { plantType, status, recommendation, imageUrl, thumbnailUrl } }
    });

    res.json({ status, recommendation, imageUrl, thumbnailUrl });
  } catch (e) {
    next(e);
  }
});

app.get("/history", authenticate, async (req, res, next) => {
  try {
    const page  = Math.max(+req.query.page || 1, 1);
    const limit = Math.min(+req.query.limit || 10, 100);
    const skip  = (page - 1) * limit;

    const [doc] = await User.aggregate([
      { $match: { _id: mongoose.Types.ObjectId(req.userId) } },
      {
        $project: {
          username: 1,
          total:    { $size: "$history" },
          history:  { $slice: [ "$history", skip, limit ] }
        }
      }
    ]);
    if (!doc) return res.status(404).json({ message: "Not found" });

    res.json({
      username: doc.username,
      history:  doc.history,
      page,
      limit,
      total: doc.total,
      pages: Math.ceil(doc.total / limit)
    });
  } catch (e) {
    next(e);
  }
});

// Error handlers
app.use((err, req, res, _) => {
  console.error(err);
  res.status(500).json({ message: err.message || "Error" });
});
process.on("uncaughtException", e => {
  console.error("Uncaught:", e);
  process.exit(1);
});
process.on("unhandledRejection", e => {
  console.error("Rejection:", e);
  process.exit(1);
});

// Start
const server = app.listen(PORT, "0.0.0.0", () => console.log(`🚀 Server on ${PORT}`));
server.on("error", e => {
  if (e.code === "EADDRINUSE") {
    console.error(`Port ${PORT} busy`);
    process.exit(1);
  }
  console.error("Server err", e);
  process.exit(1);
});
