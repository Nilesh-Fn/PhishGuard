import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Handle NaN or non-string values
    if not isinstance(text, str):
        return ''
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Keep only letters, convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    # Tokenize and remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    
    return ' '.join(tokens)

# Read CSV
df = pd.read_csv(r'C:\Users\91876\Desktop\phish guard\dataset\Assassin.csv')

# Check if 'body' column exists
if 'body' not in df.columns:
    print("⚠️ Error: 'body' column not found in the dataset!")
    print("Available columns:", df.columns)
else:
    # Apply cleaning function safely
    df['clean_text'] = df['body'].apply(clean_text)
    df.to_csv(r'C:\Users\91876\Desktop\phish guard\dataset\cleaned_emails.csv', index=False)
    print("✅ Preprocessing complete! File saved as 'cleaned_emails.csv'")

