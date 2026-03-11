
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import logging
from googletrans import Translator
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load the Gemini API key from the environment variable
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logging.error("Missing API_KEY environment variable.")
    raise Exception("Missing API_KEY environment variable. Please set it in your Render environment.")

genai.configure(api_key=API_KEY)
logging.info("Gemini API key configured successfully.")

# List available models for debugging purposes
try:
    models = genai.list_models()
    logging.info("Available models: %s", models)
except Exception as e:
    logging.error("Failed to list available models: %s", e)

# Initialize the translator and language mappings
translator = Translator()
LANGUAGE_MAP = {
    "english": "en",
    "telugu": "te",
    "hindi": "hi",
    # add more mappings as needed
}

def get_cure_recommendation(username, status, plant_type, water_frequency, disease):
    """
    Generates a fresh, personalized plant care recommendation.
    """
    greeting = f"Dear {username}," if username else ""
    prompt = f"""
{greeting}
You are a compassionate and knowledgeable plant care advisor. Based on the details provided below, please generate a personalized plant care recommendation that is completely fresh and unique each time.

If the plant appears healthy and there is no disease:
- Begin with a cheerful greeting.
- Include a **new, never-repeated fun fact** about caring for a {plant_type}.
- Recommend a maintenance fertilizer and include an online purchase link if possible.
- Use upbeat language with plant-related emojis (e.g., 🌿, 🌸, 🍃).

If the plant shows signs of disease or abnormality (e.g., '{disease}', yellow leaves, spots, or drooping):
- Begin with a gentle, empathetic concern using caution emojis (⚠️🚨).
- Provide **5 concise, numbered steps** focusing on recovery and improved care.
- Recommend at least **two specific fertilizers** (include brand names if possible) with direct online purchase links if available.
- Suggest natural remedies or adjustments in the care routine.
- Ensure the recovery advice is new and uniquely generated.

---------------------------
User-Provided Details:
- Plant Status: {status}
- Plant Type: {plant_type}
- Watering Frequency: Every {water_frequency} days
- Disease: {disease}

Additional Instructions:
- Always generate a fresh and unique recommendation that feels new each time.
- Avoid repeating phrases or identical wording from previous responses.
- Keep the response engaging, concise, and no longer than 6 text lines.
- Include fertilizer suggestions with online links where possible.
---------------------------

Generate the personalized, engaging recommendation based on the above instructions.
    """
    logging.info("Generating recommendation for plant type: %s", plant_type)
    try:
        # Use the fully qualified model name with the "models/" prefix.
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            logging.info("Recommendation generated successfully.")
            return response.text.strip()
        else:
            logging.error("Gemini API did not return a valid text response.")
            return "The Gemini API did not return a valid response. Please try again later."
    except Exception as e:
        logging.exception("Error generating recommendation:")
        return f"An error occurred while generating the recommendation: {str(e)}"

@app.route('/recommend', methods=['POST'])
def gemini_recommendation():
    logging.info("Received a recommendation request.")
    data = request.get_json()
    if not data:
        logging.error("Invalid JSON payload received.")
        return jsonify({'error': 'Invalid JSON payload'}), 400

    username = data.get('username')
    status = data.get('status')
    plant_type = data.get('plantType')
    water_frequency = data.get('waterFreq')
    language = data.get('language', "english")
    disease = data.get('disease', "none")

    missing_fields = []
    if not status:
        missing_fields.append("status")
    if not plant_type:
        missing_fields.append("plantType")
    if not water_frequency:
        missing_fields.append("waterFreq")
    if not disease:
        missing_fields.append("disease")
    if missing_fields:
        logging.error("Missing required parameters: %s", ', '.join(missing_fields))
        return jsonify({'error': f"Missing required parameter(s): {', '.join(missing_fields)}"}), 400

    recommendation = get_cure_recommendation(username, status, plant_type, water_frequency, disease)

    # Translate recommendation if needed
    if language.lower() != "english":
        dest_lang = LANGUAGE_MAP.get(language.lower(), "en")
        logging.info("Translating recommendation to language: %s", language)
        try:
            translated = translator.translate(recommendation, dest=dest_lang)
            recommendation = translated.text
            logging.info("Translation successful.")
        except Exception as e:
            logging.exception("Translation error:")
            recommendation += "\n\n(Note: Translation to your selected language failed.)"

    logging.info("Sending response back to client.")
    return jsonify({'recommendation': recommendation})

if __name__ == '__main__':
    # Run on port 5001 (adjust if needed)
    app.run(host='0.0.0.0', port=5001, debug=True)
