from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Setup ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- Preprocessing Function ---
def clean_text(text):
    if not isinstance(text, str):
        text = ''
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)      # Remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()    # Keep letters only
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

# --- Flask App ---
app = Flask(__name__)

# --- Load Model & Vectorizer with updated path ---
MODEL_PATH = r'C:\Users\91876\Desktop\phish guard\Train_model\Train_model\phishing_model.pkl'
VECTORIZER_PATH = r'C:\Users\91876\Desktop\phish guard\Train_model\Train_model\vectorizer.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email')
    if not email:
        return render_template('index.html', prediction="⚠️ No email received!", email=email, confidence=0)

    email_cleaned = clean_text(email)
    vector = vectorizer.transform([email_cleaned])
    result = model.predict(vector)[0]

    # Get confidence if supported
    confidence = 0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vector)[0]
        # find the index of the predicted class
        try:
            pred_index = list(model.classes_).index(result)
            confidence = round(probs[pred_index]*100, 2)  # percentage
        except:
            confidence = 0

    # Friendly prediction
    if str(result).lower() in ['phishing', 'spam', '1']:
        prediction_text = "⚠️ This email is likely PHISHING!"
        color_class = "phishing"
    else:
        prediction_text = "✅ This email appears LEGITIMATE."
        color_class = "legitimate"

    return render_template(
        'index.html',
        prediction=prediction_text,
        email=email,
        confidence=confidence,
        color_class=color_class
    )
# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
