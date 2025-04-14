from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not present
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set relative paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Load the trained model and vectorizer using relative paths
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Ensure they are in the 'model' directory.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Text preprocessing function
def clean_text(text):
    """Cleans and preprocesses input text for model prediction."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]

    cleaned_text = ' '.join(words)
    return cleaned_text if cleaned_text.strip() else "no_valid_words"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Input text is empty"}), 400

        # Preprocess the input text
        cleaned_text = clean_text(text)

        if cleaned_text == "no_valid_words":
            return jsonify({"error": "Input text contains no valid words"}), 400

        # Convert text to vector
        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]
        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/')
def home():
    return "Fake News Detection API is Live! Use the '/predict' endpoint to POST text."

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
