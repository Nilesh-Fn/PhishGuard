import re
import joblib
import pandas as pd

# Optional: only import heavy NLTK parts if needed
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Ensure the resources exist (silent if already downloaded)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
except Exception as e:
    # If NLTK not available, fallback to very simple preprocessing
    STOPWORDS = set()
    LEMMATIZER = None
    print("Warning: NLTK not available. Using simpler preprocessing:", e)

def clean_text(text):
    """Basic cleaning to mirror training preprocessing."""
    if not isinstance(text, str):
        text = ''
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # keep only letters, convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = []
    for w in text.split():
        if w in STOPWORDS:
            continue
        if LEMMATIZER:
            tokens.append(LEMMATIZER.lemmatize(w))
        else:
            tokens.append(w)
    return ' '.join(tokens)

# --- Load model & vectorizer safely ---
try:
    model = joblib.load('phishing_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError as e:
    raise SystemExit(f"Model or vectorizer file not found: {e}")

# Example email (replace with any text)
raw_email = """Your account is under threat. Update payment info now."""
# Preprocess like training
email_processed = clean_text(raw_email)

# Vectorize and predict
vec = vectorizer.transform([email_processed])
pred = model.predict(vec)[0]

# Try to interpret prediction using model.classes_ if available
label_to_print = None
if hasattr(model, 'classes_'):
    classes = list(model.classes_)
    # If classes are strings and include 'phishing' or 'spam', use them directly
    if any(isinstance(c, str) and c.lower() in ('phishing','spam') for c in classes):
        # pred already matches those string labels
        label_to_print = str(pred)
    else:
        # If classes look numeric like [0,1], assume 1 -> phishing
        try:
            numeric_classes = [float(c) for c in classes]
            # Map numeric prediction to text: prefer 1 -> phishing, else use class name
            if 1.0 in numeric_classes:
                label_to_print = 'Phishing' if float(pred) == 1.0 else 'Legitimate'
            else:
                label_to_print = str(pred)
        except Exception:
            label_to_print = str(pred)
else:
    # fallback: if prediction is string use it, else interpret 1 as phishing
    if isinstance(pred, str):
        label_to_print = pred
    else:
        label_to_print = 'Phishing' if pred == 1 else 'Legitimate'

# Optionally show probability if model supports it
prob_text = ""
if hasattr(model, 'predict_proba'):
    try:
        probs = model.predict_proba(vec)[0]
        # find index of predicted class and show its probability
        pred_index = list(model.classes_).index(pred)
        prob_text = f" (confidence: {probs[pred_index]:.2f})"
    except Exception:
        pass

print("Prediction:", label_to_print + prob_text)
