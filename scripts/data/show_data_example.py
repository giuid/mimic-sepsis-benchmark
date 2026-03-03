import numpy as np
import pandas as pd

# Load test data
data_path = "data/processed/test.npz"
print(f"Loading data from {data_path}...")
loaded = np.load(data_path)
data = loaded["data"]
mask = loaded["orig_mask"]

# Feature names from config
feature_names = [
    "Heart Rate", "SBP", "DBP", "Resp Rate", "SpO2", "Temp",
    "Glucose", "K+", "Na+", "Cl-", "Creatinine", "BUN",
    "WBC", "Platelets", "Hgb", "Hct", "Lactate"
]

# Select first patient
patient_idx = 0
patient_data = data[patient_idx] # (T, D)
patient_mask = mask[patient_idx] # (T, D)

T, D = patient_data.shape
print(f"\nData Shape: {data.shape} (Patients, TimeSteps, Features)")
print(f"Example Patient ({patient_idx}): {T} time steps (48h), {D} features")

# Create a DataFrame for better visualization
# We'll show the first 12 hours (time steps)
hours_to_show = 12
print(f"\n--- First {hours_to_show} Hours of Data for Patient {patient_idx} ---")
print("Values are Z-score normalized. '---' indicates missing value (mask=0).")

metrics = []
for t in range(hours_to_show):
    row = {}
    row["Time (h)"] = t
    for d in range(D):
        feat = feature_names[d]
        val = patient_data[t, d]
        m = patient_mask[t, d]
        
        if m == 1:
            row[feat] = f"{val:.2f}"
        else:
            row[feat] = "---"
    metrics.append(row)

df = pd.DataFrame(metrics)
print(df.to_string(index=False))
