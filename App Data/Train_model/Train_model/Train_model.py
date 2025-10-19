import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1️⃣ Load cleaned dataset
data = pd.read_csv(r'C:\Users\91876\Desktop\phish guard\dataset\cleaned_emails.csv')

# 2️⃣ Verify required columns exist
if 'clean_text' not in data.columns or 'label' not in data.columns:
    print("⚠️ Error: Required columns not found!")
    print("Available columns:", list(data.columns))
else:
    # 3️⃣ Drop missing or empty values
    data = data.dropna(subset=['clean_text', 'label'])
    data = data[data['clean_text'].str.strip() != '']

    # 4️⃣ Split data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['label'], test_size=0.2, random_state=42
    )

    # 5️⃣ Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # 6️⃣ Train Random Forest model
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train_vect, y_train)

    # 7️⃣ Evaluate performance
    predictions = model.predict(X_test_vect)
    print("\n✅ Model Evaluation:\n")
    print(classification_report(y_test, predictions))

    # 8️⃣ Save model and vectorizer
    joblib.dump(model, 'phishing_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("\n💾 Model saved as 'phishing_model.pkl'")
    print("💾 Vectorizer saved as 'vectorizer.pkl'")
