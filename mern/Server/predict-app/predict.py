
import os
import io
import re
import tempfile
import traceback
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

from gradio_client import Client, handle_file

# ——— Configuration —————————————
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # Disable GPU (if you like)
PORT               = int(os.environ.get("PORT", 5002))
HEALTHY_THRESHOLD  = 0.5

# Replace with your actual HF Space names:
BINARY_SPACE       = "vittamraj/predict-binary-api"
MULTI_SPACE        = "vittamraj/predict-MultiClass-api"

# ——— Logging & Flask setup —————————
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Instantiate HF Space clients
binary_client = Client(BINARY_SPACE)
multi_client  = Client(MULTI_SPACE)

# ——— Image pre-processing —————————
def dynamic_crop(img: Image.Image) -> Image.Image:
    gray     = img.convert("L")
    inverted = ImageOps.invert(gray)
    bw       = inverted.point(lambda x: 0 if x < 50 else 255, "1")
    bbox     = bw.getbbox()
    return img.crop(bbox) if bbox else img

def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = dynamic_crop(img)

    # center-crop to square
    w, h   = img.size
    m      = min(w, h)
    left   = (w - m) // 2
    top    = (h - m) // 2
    img    = img.crop((left, top, left + m, top + m))

    # resize to 224×224
    return img.resize((224, 224), Image.LANCZOS)

# ——— Parse “Prediction: Class (Confidence: 0.95)” strings ——
def parse_result(text: str):
    match = re.search(r"Prediction:\s*([\w_]+)\s*\(Confidence:\s*([\d\.]+)\)", text)
    if match:
        return match.group(1), float(match.group(2))
    return text, 0.0

# ——— Health check ————————————————
@app.route("/", methods=["GET", "HEAD"])
def health():
    return "", 200

# ——— Prediction endpoint ——————————
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # CORS preflight
        return jsonify({}), 200

    # Optional form fields
    plant_type = request.form.get("plantType", "unknown")
    water_freq = request.form.get("waterFreq", "unknown")
    language   = request.form.get("language", "english")

    # Load image bytes
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.data:
        image_bytes = request.data
    else:
        return jsonify({"error": "No image provided"}), 400

    # Preprocess
    try:
        img = preprocess_image(image_bytes)
    except Exception as e:
        logging.error("Preprocessing failed: " + str(e))
        return jsonify({"error": f"Image preprocessing failed: {e}"}), 500

    # Write to temp file for gradio_client.handle_file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        # 1) Binary API
        bin_res = binary_client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )
        logging.info(f"Binary API → {bin_res}")

        if isinstance(bin_res, str):
            pred, conf = parse_result(bin_res)
        else:
            pred = bin_res.get("prediction") or bin_res.get("Prediction", "")
            conf = float(bin_res.get("confidence", 0))

        if pred == "Healthy" and conf >= HEALTHY_THRESHOLD:
            status         = "Healthy_plants"
            recommendation = f"Your {plant_type} appears healthy. Continue regular care."
        else:
            # 2) Multi-class API for refined label
            multi_res = multi_client.predict(
                image=handle_file(tmp_path),
                api_name="/predict"
            )
            logging.info(f"Multiclass API → {multi_res}")

            if isinstance(multi_res, str):
                status, _ = parse_result(multi_res)
            else:
                status = multi_res.get("prediction") or multi_res.get("Prediction", "Unknown")

            recommendation = (
                f"Your {plant_type} shows signs of {status}. "
                f"Adjust care routine; water every {water_freq} days. (Language: {language})"
            )

        return jsonify({ "status": status, "recommendation": recommendation })

    except Exception:
        logging.error("Prediction error:\n" + traceback.format_exc())
        return jsonify({"error": "Prediction failed"}), 500

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
