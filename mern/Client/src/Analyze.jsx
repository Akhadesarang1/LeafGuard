
import axios from "axios";
import { motion } from "framer-motion";
import React, { useEffect, useRef, useState } from "react";
import {
  FaArrowLeft,
  FaCamera,
  FaCheckCircle,
  FaExclamationTriangle,
  FaImages,
  FaSpinner,
} from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "environment",
};

const containerVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

const Analyze = () => {
  const navigate = useNavigate();

  // Form states
  const [image, setImage] = useState(null);
  const [plantType, setPlantType] = useState("neem");
  const [waterFreq, setWaterFreq] = useState("1");
  const [language, setLanguage] = useState("english");

  // Result & status states
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Preview state to show the uploaded or captured image
  const [preview, setPreview] = useState(null);

  // Camera state
  const [useCamera, setUseCamera] = useState(false);
  const webcamRef = useRef(null);

  // Clean up preview URL when component unmounts or file changes
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
    } else {
      setPreview(null);
    }
  };

  // Helper: Convert data URL to a File object
  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  // Capture image from webcam
  const handleCapture = () => {
    if (webcamRef.current) {
      const screenshot = webcamRef.current.getScreenshot();
      if (screenshot) {
        const capturedFile = dataURLtoFile(screenshot, "captured.jpg");
        setImage(capturedFile);
        setPreview(screenshot);
        setUseCamera(false);
      }
    }
  };

  // Call your Node server on Render to analyze
  const handleAnalyze = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!image) {
      setError("Please upload or capture an image.");
      return;
    }

const token = localStorage.getItem("token");
    if     (!token) {
      setError("You must be logged in to analyze.");
      return;
    }

    // Prepare form data for the Node /analyze endpoint
    const formData = new FormData();
    formData.append("image", image);
    formData.append("plantType", plantType);
    formData.append("waterFreq", waterFreq);
    formData.append("language", language);

    try {
      setLoading(true);
      // Use your Node backend URL
      const response = await axios.post(
        "https://backend-lj86.onrender.com/analyze",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${token}`,
          },
        }
      );
      // The backend returns { prediction, recommendation }
      const { prediction, recommendation } = response.data;
      setResult({ prediction: status, recommendation });
    } catch (err) {
      setError(
        err.response?.data?.error || "An error occurred while analyzing."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="relative min-h-screen font-sans"
      style={{
        backgroundImage: "url('/assets/analyze-bg.jpg')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      {/* Responsive Centered Content */}
      <div className="relative z-10 flex flex-col items-center justify-center p-4 sm:p-6 min-h-screen">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="bg-white/90 backdrop-blur-md p-6 sm:p-8 rounded-xl shadow-xl w-full max-w-lg mx-auto"
        >
          {/* Back Button */}
          <motion.button
            onClick={() => navigate("/")}
            whileHover={{ scale: 1.1, color: "#f87171" }}
            whileTap={{ scale: 0.95 }}
            className="absolute top-4 left-4 text-white transition-colors"
          >
            <FaArrowLeft size={20} />
          </motion.button>

          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center text-3xl sm:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-500 mb-6"
          >
            Analyze Your Plant
          </motion.h2>

          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded"
            >
              {error}
            </motion.div>
          )}

          <form onSubmit={handleAnalyze} className="space-y-6">
            {/* Plant Type Dropdown */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Plant Type:
              </label>
              <motion.select
                whileHover={{ scale: 1.05 }}
                value={plantType}
                onChange={(e) => setPlantType(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              >
                <option value="neem">Neem</option>
                <option value="guava">Guava</option>
                <option value="money plant">Money Plant</option>
                <option value="tomato">Tomato</option>
              </motion.select>
            </div>

            {/* Water Frequency Dropdown */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Water Frequency (days):
              </label>
              <motion.select
                whileHover={{ scale: 1.05 }}
                value={waterFreq}
                onChange={(e) => setWaterFreq(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              >
                {[...Array(7)].map((_, i) => (
                  <option key={i + 1} value={i + 1}>
                    {i + 1}
                  </option>
                ))}
              </motion.select>
            </div>

            {/* Language Dropdown */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Language:
              </label>
              <motion.select
                whileHover={{ scale: 1.05 }}
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              >
                <option value="english">English</option>
                <option value="telugu">Telugu</option>
                <option value="hindi">Hindi</option>
              </motion.select>
            </div>

            {/* Image Upload & Camera Options */}
            <div>
              <label className="block text-gray-700 font-medium mb-1">
                Upload or Capture Image:
              </label>
              <div className="flex flex-col sm:flex-row gap-4">
                {/* Upload Option */}
                <motion.div whileHover={{ scale: 1.05 }} className="flex-1">
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer flex items-center justify-center p-2 border border-gray-300 rounded-md text-center hover:bg-blue-50 transition"
                  >
                    <span className="truncate max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap flex items-center justify-center">
                      {image ? image.name : (<><FaImages className="mr-2" /> Gallery</>)}
                    </span>
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </motion.div>
                {/* Camera Option */}
                <motion.div whileHover={{ scale: 1.05 }} className="flex-1">
                  <motion.button
                    type="button"
                    onClick={() => setUseCamera(!useCamera)}
                    className="w-full flex items-center justify-center p-2 border border-gray-300 rounded-md text-center hover:bg-blue-50 transition"
                  >
                    <FaCamera className="mr-2" /> Use Camera
                  </motion.button>
                </motion.div>
              </div>

              {/* Display Camera if toggled */}
              {useCamera && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 relative"
                >
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="w-full rounded-md shadow-md"
                  />
                  <div className="flex justify-between mt-2">
                    <motion.button
                      type="button"
                      onClick={handleCapture}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="flex-1 mr-2 py-2 px-4 bg-green-500 hover:bg-green-600 text-white rounded transition"
                    >
                      Capture Photo
                    </motion.button>
                    <motion.button
                      type="button"
                      onClick={() => setUseCamera(false)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="flex-1 ml-2 py-2 px-4 bg-red-500 hover:bg-red-600 text-white rounded transition"
                    >
                      Cancel
                    </motion.button>
                  </div>
                </motion.div>
              )}

              {/* Image Preview */}
              {preview && (
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="mt-4 flex justify-center"
                >
                  <img
                    src={preview}
                    alt="Preview"
                    className="max-h-40 object-cover rounded-md shadow-md"
                  />
                </motion.div>
              )}
            </div>

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="w-full py-2 px-4 bg-blue-500 hover:bg-blue-600 text-white font-semibold rounded-full transition flex items-center justify-center"
            >
              {loading ? (
                <>
                  <FaSpinner className="animate-spin mr-2" /> Diagnosing...🏃‍♂
                </>
              ) : (
                "Diagnose"
              )}
            </motion.button>
          </form>

          {/* Display Result */}
          {result && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={`mt-6 p-4 rounded-md border-2 shadow-lg transition-all duration-500 ease-out ${
                result.prediction === "Healthy_Plants"
                  ? "border-green-400 bg-green-100"
                  : "border-red-400 bg-red-100"
              }`}
            >
              <div className="flex items-center mb-2">
                {result.prediction === "Healthy_Plants" ? (
                  <FaCheckCircle className="text-green-500 mr-2" />
                ) : (
                  <FaExclamationTriangle className="text-red-500 mr-2" />
                )}
                <h3 className="text-xl font-bold text-gray-800">
                  {result.prediction === "Healthy_Plants"
                    ? "Healthy"
                    : `Status: ${result.prediction}`}
                </h3>
              </div>
              <p className="text-gray-700 whitespace-pre-wrap leading-relaxed mb-2">
                {result.recommendation || "No recommendation provided."}
              </p>
              {result.prediction !== "Healthy_Plants" && (
                <motion.button
                  onClick={() => navigate(`/cure-links/${result.prediction}`)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="mt-4 w-full py-2 px-4 bg-purple-500 hover:bg-purple-600 text-white font-semibold rounded-full transition"
                >
                  Browse Cure Links
                </motion.button>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default Analyze;
