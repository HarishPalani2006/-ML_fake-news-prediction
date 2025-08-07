from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
# Load model and vectorizer
model = pickle.load(open(r'E:\project\fake news\rf_model.pkl', 'rb'))
vectorizer = pickle.load(open(r'E:\project\fake news\vectorizer.pkl', 'rb'))


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        news_text = request.form['news']
        if news_text.strip():
            cleaned_text = clean_text(news_text)  # Clean like training
            transformed = vectorizer.transform([cleaned_text])
            result = model.predict(transformed)[0]
            prediction = "🟥 FAKE news!" if result == 'fake' else "🟩 REAL news!"
        else:
            prediction = "⚠️ Please enter some text."
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
