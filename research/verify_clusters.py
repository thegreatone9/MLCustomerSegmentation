"""Verify that dynamic cluster mapping works correctly."""
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("artifacts/model_trainer/model.joblib")
scaler = joblib.load("artifacts/data_transformation/scaler.joblib")

# Import the fixed pipeline
import sys
sys.path.insert(0, "src")
from customerSegmentation.pipeline.prediction import PredictionPipeline

pipeline = PredictionPipeline()

# Show cluster map
centers_original = scaler.inverse_transform(model.cluster_centers_)
print("=" * 70)
print("CLUSTER CENTER → SEGMENT MAPPING (Dynamic)")
print("=" * 70)
print(f"\n{'Cluster':<10} {'Income (k$)':<15} {'Spending':<15} {'Mapped Name':<25}")
print("-" * 70)
for cid, (income, spending) in enumerate(centers_original):
    name = pipeline.cluster_map[cid]['name']
    print(f"{cid:<10} ${income:<14.1f} {spending:<15.1f} {name}")

# Spot checks
print("\n" + "=" * 70)
print("PREDICTION SPOT CHECKS")
print("=" * 70)
test_cases = [
    ([15, 10], "Careful Customers"),
    ([50, 50], "Standard Customers"),
    ([100, 90], "Target Customers"),
    ([20, 80], "Careless Customers"),
    ([120, 10], "Sensible Customers"),
]

all_pass = True
for inputs, expected in test_cases:
    result = pipeline.predict([inputs])
    status = "✅" if result['name'] == expected else "❌"
    if status == "❌":
        all_pass = False
    print(f"\nInput: Income=${inputs[0]}k, Spending={inputs[1]}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result['name']} {status}")

print("\n" + "=" * 70)
print(f"RESULT: {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
print("=" * 70)
