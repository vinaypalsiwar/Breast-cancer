from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Load ML files
model = pickle.load(open("cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    feature_names = [
        "Tumor Radius",
        "Cell Texture Irregularity",
        "Tumor Boundary Length",
        "Tumor Area",
        "Tumor Edge Smoothness",
        "Tumor Cell Density",
        "Tumor Concavity"
    ]

    # ---------------- Patient History ----------------
    age = request.form['age']
    family_history = request.form['family_history']
    previous_cancer = request.form['previous_cancer']

    history_text = (
        f"Patient Age: {age}, "
        f"Family History: {family_history}, "
        f"Previous Cancer: {previous_cancer}"
    )

    # ---------------- ML Features ----------------
    features = [
        float(request.form['tumor_radius']),
        float(request.form['cell_texture_irregularity']),
        float(request.form['tumor_boundary_length']),
        float(request.form['tumor_area']),
        float(request.form['tumor_edge_smoothness']),
        float(request.form['tumor_cell_density']),
        float(request.form['tumor_concavity'])
    ]

    features = np.array(features).reshape(1, -1)
    features = imputer.transform(features)
    features = scaler.transform(features)

    # ---------------- Prediction ----------------
    prob = model.predict_proba(features)[0]
    malignant_prob = prob[0] * 100
    benign_prob = prob[1] * 100

    # ---------------- Recommendations ----------------
    if malignant_prob > 60:
        result = f"Malignant (Cancer Detected) — {malignant_prob:.2f}% confidence"

        recommendations = [
            "Consult an oncologist immediately",
            "Treatment options may include surgery, chemotherapy, radiation, or hormone therapy",
            "Follow biopsy and staging investigations as advised by doctors"
        ]

        precautions = [
            "Avoid self-medication",
            "Maintain proper nutrition and mental well-being",
            "Attend all follow-up appointments"
        ]

        awareness = [
            "Early treatment significantly improves survival rates",
            "Family support and counseling are important",
            "Regular monitoring is essential"
        ]
    else:
        result = f"Benign (No Cancer) — {benign_prob:.2f}% confidence"

        recommendations = [
            "No immediate treatment required",
            "Continue routine medical checkups"
        ]

        precautions = [
            "Perform regular breast self-examinations",
            "Maintain a healthy lifestyle",
            "Schedule periodic clinical screenings"
        ]

        awareness = [
            "Benign tumors can change over time",
            "Family history increases future risk",
            "Early detection saves lives"
        ]

    # ---------------- Feature Influence (Top 3) ----------------
    coefficients = model.coef_[0]
    influence_scores = coefficients * features[0]
    top_idx = np.argsort(np.abs(influence_scores))[::-1][:3]

    top_labels = [feature_names[i] for i in top_idx]
    top_values = [abs(influence_scores[i]) for i in top_idx]

    chart_colors = [
        "rgba(220,53,69,0.8)" if influence_scores[i] > 0 else "rgba(40,167,69,0.8)"
        for i in top_idx
    ]

    explanation_text = "Top influencing factors: " + ", ".join(top_labels)

    # Save report data for PDF
    app.config["report_data"] = {
        "result": result,
        "history": history_text,
        "recommendations": recommendations,
        "precautions": precautions,
        "awareness": awareness
    }

    return render_template(
        "index.html",
        prediction_text=result,
        explanation_text=explanation_text,
        history_text=history_text,
        recommendations=recommendations,
        precautions=precautions,
        awareness=awareness,
        chart_labels=json.dumps(top_labels),
        chart_values=json.dumps(top_values),
        chart_colors=json.dumps(chart_colors),
        show_pdf=True
    )

@app.route("/download_report")
def download_report():
    data = app.config.get("report_data")

    file_path = "breast_cancer_report.pdf"
    pdf = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("<b>Breast Cancer Prediction Report</b>", styles['Title']),
        Paragraph(data["result"], styles['Normal']),
        Paragraph(data["history"], styles['Normal']),
        Paragraph("<b>Recommendations</b>", styles['Heading2'])
    ]

    for r in data["recommendations"]:
        content.append(Paragraph(f"- {r}", styles['Normal']))

    content.append(Paragraph("<b>Precautions</b>", styles['Heading2']))
    for p in data["precautions"]:
        content.append(Paragraph(f"- {p}", styles['Normal']))

    content.append(Paragraph("<b>Awareness</b>", styles['Heading2']))
    for a in data["awareness"]:
        content.append(Paragraph(f"- {a}", styles['Normal']))

    pdf.build(content)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
