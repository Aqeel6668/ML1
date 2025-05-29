import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']


def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])  # ham=0, spam=1


X = df['cleaned_text']
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
