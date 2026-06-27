from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    data = vectorizer.transform([message])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Spam Email"
    else:
        result = "Not Spam"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)