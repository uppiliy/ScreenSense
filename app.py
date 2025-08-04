from flask import Flask, render_template, request
from collections import defaultdict
import os
import pandas as pd

# ─── Feature lists ───
numeric_features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
categorical_features = ['Gender', 'Primary_Device', 'Health_Impacts', 'Urban_or_Rural']

# ─── Human-friendly feature names ───
feature_aliases = {
    "Age": "Age",
    "Avg_Daily_Screen_Time_hr": "Average Daily Screen Time (hours)",
    "Educational_to_Recreational_Ratio": "Educational vs. Recreational Usage Ratio",
    "Gender": "Gender",
    "Primary_Device": "Main Device Used",
    "Health_Impacts": "Reported Health Impacts",
    "Urban_or_Rural": "Living Environment"
}

# ─── Placeholders for heavy resources ───
app = Flask(__name__)
model = None
preproc = None
explainer = None
flan_generator = None

def consolidate_features(features):
    merged = defaultdict(float)
    for name, val in features:
        merged[name] += val
    return list(merged.items())

def generate_nl_explanation(top_features, feature_values):
    top_features = consolidate_features(top_features)
    top_features = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)[:3]

    prompt = (
        "You're a kind assistant helping a concerned parent understand their child's screen time habits. "
        "Below are the most important factors and how they influenced the result:\n\n"
    )
    any_increase = False
    for name, contrib in top_features:
        pretty = feature_aliases.get(name, name)
        value = feature_values.get(name, "N/A")
        direction = "increased" if contrib > 0 else "reduced"
        if contrib > 0:
            any_increase = True
        prompt += f"- {pretty} was {value}, which {direction} the chances of exceeding healthy screen time.\n"

    prompt += (
        "\nPlease explain this in simple and caring language (2–3 sentences). "
        "Avoid repeating feature names."
    )
    if any_increase:
        prompt += " Also, suggest one gentle tip to help improve the child's screen time habits."

    out = flan_generator(prompt, max_new_tokens=200)
    return out[0]["generated_text"].strip()

def generate_user_feedback(explanation_text):
    prompt = (
        f"Here’s an explanation of why a child's screen time prediction was made:\n\n"
        f"{explanation_text}\n\n"
        "Now, as a friendly parenting coach, give 2–3 distinct, concise, and non-repetitive suggestions"
        " to help the parent support healthier screen time habits. Do not repeat any phrase."
    )
    out = flan_generator(prompt, max_new_tokens=150)
    return out[0]["generated_text"].strip()

import re
def clean_feedback(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, result = set(), []
    for s in parts:
        norm = s.lower().strip()
        if norm not in seen:
            seen.add(norm)
            result.append(s)
    return " ".join(result)

@app.before_first_request
def init_resources():
    """
    Load model, SHAP explainer, and Flan-T5 pipeline only once—
    after Gunicorn has bound to the port.
    """
    global model, preproc, explainer, flan_generator

    # Lazy‐import heavy libs
    import joblib
    import shap
    from transformers import pipeline

    # ─── 1) Load your trained pipeline ───
    model = joblib.load("logistic_model_pipeline.pkl")
    preproc = model.named_steps['preproc']
    clf     = model.named_steps['clf']

    # ─── 2) Prepare SHAP explainer ───
    df_bg = pd.read_csv("Indian_Kids_Screen_Time.csv")[numeric_features + categorical_features]
    df_bg_sample = df_bg.sample(n=100, random_state=0)
    X_bg_processed = preproc.transform(df_bg_sample)
    explainer = shap.LinearExplainer(clf, X_bg_processed, feature_perturbation="interventional")

    # ─── 3) GenAI explainer (Flan-T5) ───
    flan_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

@app.route("/", methods=["GET", "POST"])
def predict():
    result = explanation = feedback = None

    if request.method == "POST":
        # 1. Gather inputs
        fv = {
            "Age": float(request.form["age"]),
            "Avg_Daily_Screen_Time_hr": float(request.form["screen_time"]),
            "Educational_to_Recreational_Ratio": float(request.form["ratio"]),
            "Gender": request.form["gender"],
            "Primary_Device": request.form["device"],
            "Health_Impacts": request.form["health"],
            "Urban_or_Rural": request.form["location"]
        }
        input_df = pd.DataFrame([fv])

        # 2. Make prediction
        pred = model.predict(input_df)[0]
        result = "Exceeded Limit" if pred else "Within Limit"

        # 3. Compute SHAP values
        X_proc = preproc.transform(input_df)
        shap_vals = explainer.shap_values(X_proc)
        sample_contribs = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]

        # 4. Map back to original features
        cat_ohe    = preproc.named_transformers_['cat']['onehot']
        cat_names  = cat_ohe.get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_names)
        raw_feats  = list(zip(feature_names, sample_contribs))

        # 5. Group one-hot columns
        grouped = defaultdict(float)
        for name, val in raw_feats:
            base = name.split('_')[0] if name in cat_names else name
            grouped[base] += val
        top_feats = list(grouped.items())

        # 6. Generate explanation & feedback
        explanation = generate_nl_explanation(top_feats, fv)
        raw_fb      = generate_user_feedback(explanation)
        feedback    = clean_feedback(raw_fb)

    return render_template("form.html",
                           result=result,
                           explanation=explanation,
                           feedback=feedback)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
