import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# 1. Load the dataset
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# 2. Label the data
fake['label'] = 'fake'
true['label'] = 'true'

# 3. Merge and shuffle
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)
df = df[['text', 'label']]

# 4. Preprocess the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)

# 5. Split into features and labels
x = df['text']
y = df['label']

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 7. Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# 9. Save the model and vectorizer
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
