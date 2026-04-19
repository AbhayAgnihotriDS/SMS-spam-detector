from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("mnb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({
                "prediction": "Enter valid text",
                "confidence": 0,
                "spam_prob": 0,
                "not_spam_prob": 0
            })

        vector = vectorizer.transform([text])

        # Prediction
        pred = model.predict(vector)[0]

        # 🔥 DEFINE HERE (IMPORTANT)
        probs = model.predict_proba(vector)[0]

        spam_prob = round(float(probs[1]) * 100, 2)
        not_spam_prob = round(float(probs[0]) * 100, 2)

        confidence = max(spam_prob, not_spam_prob)

        result = "Spam 🚫" if pred == 1 else "Not Spam ✅"

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "spam_prob": spam_prob,
            "not_spam_prob": not_spam_prob
        })

    except Exception as e:
        print("ERROR:", e)

        # ✅ IMPORTANT: define fallback values here too
        return jsonify({
            "prediction": "Server error",
            "confidence": 0,
            "spam_prob": 0,
            "not_spam_prob": 0
        })

if __name__ == "__main__":
    app.run(debug=True)