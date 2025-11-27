import os
import json
import subprocess
import numpy as np
from flask import Flask, render_template, request, jsonify
from customerSegmentation.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    """Render the landing page with prediction form."""
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def training():
    """Trigger the full ML training pipeline."""
    result = subprocess.run(
        ["python", "main.py"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return "Training Successful!"
    else:
        return f"Training Failed:\n{result.stderr}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict customer segment from form input."""
    try:
        annual_income = float(request.form['annual_income'])
        spending_score = float(request.form['spending_score'])

        pipeline = PredictionPipeline()
        result = pipeline.predict([[annual_income, spending_score]])

        return render_template('results.html',
                               prediction=result,
                               annual_income=annual_income,
                               spending_score=spending_score)

    except Exception as e:
        print(f'Prediction error: {e}')
        return f'Error: {e}', 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Return model evaluation metrics as JSON."""
    metrics_path = "artifacts/model_evaluation/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No metrics found. Train the model first."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
