Automated Insulin Delivery System — Glucose Prediction using Temporal Fusion Transformer (TFT)

This project implements a machine learning–based glucose prediction system designed to assist in Automated Insulin Delivery (AID) for Type 1 Diabetes patients.
Using real patient data from the OhioT1DM dataset, the model predicts future blood glucose levels based on historical glucose, insulin, and meal information.

The system integrates multiple deep learning architectures — GRU, Transformer, and the final Temporal Fusion Transformer (TFT) — and compares their performance using both statistical and clinical metrics.

📚 Table of Contents

Overview

Key Features

Dataset

System Workflow

Model Architectures

Evaluation Metrics

Results Summary

Error Analysis

Setup Instructions

Repository Structure

Future Improvements

Contributors

🧠 Overview

The goal of this project is to accurately predict a patient’s blood glucose level 30–60 minutes ahead and integrate this predictive capability into an automated insulin delivery loop.
This helps in preventing hypoglycemia and hyperglycemia by adjusting insulin dosage in real time.

The project explores multiple models:

GRU (Baseline) — Sequential model capturing short-term dependencies.

Transformer (Intermediate) — Attention-based model for long-range temporal patterns.

TFT (Final Model) — Combines LSTM + attention + variable selection + quantile forecasting for interpretability and robustness.

⚙️ Key Features

📈 Real-time blood glucose forecasting using deep learning

🔍 Temporal pattern learning from meal, insulin, and glucose histories

📊 Clinical-grade evaluation via Clarke Error Grid Analysis

💬 Explainable forecasts via attention and variable importance

🧮 Ensemble training and model checkpointing with PyTorch Lightning

☁️ Scalable design for cloud deployment and integration with IoT glucose sensors

📂 Dataset

Source: OhioT1DM dataset (open-source clinical dataset)

Files: 12 XML files (559-ws-training.xml, 563-ws-testing.xml, etc.)

Patients: 6 Type 1 diabetic individuals

Sampling: 5-minute glucose, insulin, and carbohydrate intake logs

Sensors: Continuous Glucose Monitor (CGM), Insulin pump, Meal entries

Data Fields Extracted:

Glucose (mg/dL)

Insulin bolus (U)

Carbohydrates (grams)

Timestamp

Derived features — glucose velocity, rolling mean/std, time since meal/bolus

🔄 System Workflow
1️⃣ XML Parsing → Extract glucose, insulin, and meal events
2️⃣ Data Alignment → Resample at 5-min intervals
3️⃣ Feature Engineering → Rolling stats, velocity, event lags
4️⃣ Scaling → Normalization of real-valued features
5️⃣ Dataset Creation → TimeSeriesDataSet for TFT
6️⃣ Model Training → GRU / Transformer / TFT
7️⃣ Evaluation → MAE, RMSE, Clarke Error Grid
8️⃣ Visualization → Predictions, Error Zones, Clinical Insights

🧩 Model Architectures
1. GRU (Gated Recurrent Unit)

Simple recurrent model capturing short-term glucose trends

Limitation: Poor at modeling meal/insulin dependencies

MAE: 28.4 mg/dL | RMSE: 35.2 mg/dL | A+B Zones: 52.6%

2. Transformer

Uses self-attention to capture long-term dependencies

Improved temporal awareness compared to GRU

MAE: 23.8 mg/dL | RMSE: 30.4 mg/dL | A+B Zones: 61.8%

3. Temporal Fusion Transformer (TFT)

Combines LSTM encoder-decoder + attention + variable selection

Learns temporal dependencies and feature importance dynamically

MAE: 14.28 mg/dL | RMSE: 20.27 mg/dL | A+B Zones: 70.2%

📏 Evaluation Metrics
Metric	Description
MAE	Mean Absolute Error — measures average deviation
RMSE	Root Mean Squared Error — penalizes large deviations
Clarke Error Grid	Evaluates clinical safety of predictions
A+B Zone Accuracy	% of predictions within clinically acceptable range
🧪 Results Summary
Model	MAE	RMSE	A+B Zones (%)
GRU	28.47	35.19	52.6
Transformer	23.82	30.44	61.8
TFT (Final)	14.28	20.27	70.2

🩺 Clinical Interpretation:

TFT achieved the highest medical safety, with ~70% of predictions in no-risk zones.

The model effectively handled missing event data and maintained stability across patient profiles.

🔍 Error Analysis

GRU: Over-smoothed predictions; struggled during rapid glucose fluctuations.

Transformer: Better temporal learning; mild overfitting on limited data.

TFT: Robust, interpretable, and generalized better; reduced bias and variance.

Clarke Error Grid visualization shows:

GRU predictions widely dispersed.

Transformer predictions closer to the diagonal.

TFT predictions tightly clustered around the line of identity.

🛠️ Setup Instructions
Prerequisites

Python 3.10+

Install dependencies:

pip install -r requirements.txt

Running the Pipeline

Clone the repo:

git clone https://github.com/<your-username>/Glucose-Prediction-TFT.git
cd Glucose-Prediction-TFT


Launch the notebook:

jupyter notebook Final_ML_Project.ipynb


Or run end-to-end:

python main_pipeline.py

Output Files

best_model.ckpt — trained TFT model

glucose_scaler.joblib — target scaler

real_features_scaler.joblib — feature scaler

Clarke Error Grid plots and evaluation report

📁 Repository Structure
📦 Glucose-Prediction-TFT/
 ┣ 📄 Final_ML_Project.ipynb          # Main notebook
 ┣ 📄 GRU_Glucose_Prediction.ipynb    # GRU baseline
 ┣ 📄 Transformer_Model.ipynb         # Transformer model
 ┣ 📄 install_dependencies.py
 ┣ 📄 glucose_scaler.joblib
 ┣ 📄 real_features_scaler.joblib
 ┣ 📄 best_model.ckpt
 ┣ 📄 README.md
 ┣ 📂 lightning_logs/
 ┣ 📂 dataset/                        # XML patient data
 ┗ 📊 results/                        # Graphs & error plots

🚀 Future Improvements

Integrate real-time CGM and insulin pump APIs

Implement TFT with attention visualization dashboard

Extend to hybrid cloud + embedded inference

Incorporate federated learning for patient privacy (HIPAA-compliant)

👩‍💻 Contributors

Project Lead: Akshaa
Guided by: [Dr. Jyoti Madake (Professor, VIT Pune)]
Technologies: PyTorch Lightning, Pandas, Matplotlib, Scikit-learn
