import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# 2. Keep only required columns
df = df[['title','description','fraudulent']]
df = df.dropna()

# 3. Combine title + description
df['text'] = df['title'] + " " + df['description']
X = df['text']
y = df['fraudulent']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train model
clf = LogisticRegression(max_iter=300)
clf.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# 8. Save model + vectorizer
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model training complete!")
