import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")


print("Loading vectorizer...")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("vectorizer loaded successfully...")