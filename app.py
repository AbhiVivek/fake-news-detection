from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load(r"C:\Users\vivek_kandikuppa\Desktop\ECE\Fake_Profile_Detection\backend\model\fake_news_model.pkl")
vectorizer = joblib.load(r"C:\Users\vivek_kandikuppa\Desktop\ECE\Fake_Profile_Detection\backend\model\vectorizer.pkl", mmap_mode='r')

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    cleaned_text = ' '.join(words)
    
    return cleaned_text if cleaned_text.strip() else None  # Return None if text becomes empty

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Input text is empty"}), 400

        # Preprocess the input text
        cleaned_text = clean_text(text)
        if cleaned_text is None:
            return jsonify({"error": "Input text contains no valid words"}), 400

        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]
        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
