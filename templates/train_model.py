# %%
'''from google.colab import files
uploaded=files.upload()'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("mail_data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Convert labels
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Convert text to lowercase
df["Message"] = df["Message"].str.lower()

# Features and labels
X = df["Message"]
y = df["Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Train model
log = LogisticRegression(max_iter=1000)

log.fit(X_train, y_train)

# Predict
y_pred = log.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(log, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model saved successfully!")


