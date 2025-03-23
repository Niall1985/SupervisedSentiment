import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

with open("training_data.json", "r", encoding = "utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df["sentiment"] = df["sentiment"].map(sentiment_map)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r"[^\w\s]", "", text)  
    words = word_tokenize(text)  
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization
    return " ".join(words)

df["clean_review"] = df["review"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_review"]).toarray()
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Support Vector Machine (SVM)
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# prints of the accuracy and classification report
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("Naïve Bayes:\n", classification_report(y_test, y_pred_nb))
print("SVM:\n", classification_report(y_test, y_pred_svm))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))


# comparisn of the model accuracies
accuracies = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Naïve Bayes": accuracy_score(y_test, y_pred_nb),
    "SVM": accuracy_score(y_test, y_pred_svm),
    "Random Forest": accuracy_score(y_test, y_pred_rf)
}

# Convert to DataFrame for visualization
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])

# Plot results
sns.barplot(x="Model", y="Accuracy", data=acc_df)
plt.title("Comparison of Supervised Learning Algorithms for Sentiment Analysis")
plt.xticks(rotation=30)
plt.show()
