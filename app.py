from flask import Flask, request, jsonify, render_template
import joblib
import re
from urllib.parse import urlparse

# Load ML model and vectorizer
clf = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# --- Rule-based checks ---
def rule_based_checks(text):
    red_flags = []

    # 1. Payment / Fee requests
    if re.search(r"(pay|deposit|registration fee|processing fee|send money)", text, re.IGNORECASE):
        red_flags.append(("Mentions payment/fees", 20))

    # 2. Free email domains
    if re.search(r"[a-zA-Z0-9._%+-]+@(gmail|yahoo|hotmail|outlook)\.com", text, re.IGNORECASE):
        red_flags.append(("Uses free email domain", 15))

    # 3. Unrealistic salary
    if re.search(r"(50k|100k|200k|per day|daily income|guaranteed income)", text, re.IGNORECASE):
        red_flags.append(("Unrealistic high salary", 25))

    # 4. Urgency / pressure
    if re.search(r"(urgent|apply fast|limited slots|only today)", text, re.IGNORECASE):
        red_flags.append(("Urgency/pressure tactic", 10))

    # 5. Shortened / suspicious links
    urls = re.findall(r"(https?://[^\s]+)", text)
    for url in urls:
        if re.search(r"(bit\.ly|tinyurl|goo\.gl|shorturl|ow\.ly)", url):
            red_flags.append(("Suspicious shortened link", 20))
        # Optional: highlight if domain looks suspicious
        domain = urlparse(url).netloc
        if domain and re.search(r"(free|cheap|earn|money|job)", domain, re.IGNORECASE):
            red_flags.append((f"Suspicious domain in URL: {domain}", 15))

    return red_flags

# --- Risk scoring ---
def calculate_final_score(ml_prob, rules):
    score = ml_prob * 100  # ML base
    for _, weight in rules:
        score += weight
    return min(score, 100)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Rule-based detection
    rules = rule_based_checks(text)

    # ML prediction
    X = vectorizer.transform([text])
    prob = clf.predict_proba(X)[0][1]
    ml_result = "Fake/Scam" if clf.predict(X)[0] == 1 else "Legit"

    # Final risk score
    final_score = calculate_final_score(prob, rules)
    decision = "⚠️ Scam Likely" if final_score >= 60 else "✅ Likely Legit"

    result = {
        "final_decision": decision,
        "risk_score": final_score,
        "ml_prediction": ml_result,
        "ml_confidence": round(prob * 100, 2),
        "rule_based_flags": [r[0] for r in rules]
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
