#  Disease Prediction System

An AI-powered medical diagnosis assistant built with Python, Scikit-learn, and Streamlit. Users can select symptoms and receive ranked disease predictions with confidence scores and symptom importance analysis.

---

##  Project Structure

```
disease_prediction/
├── app.py                          # Main Streamlit application
├── data/
│   ├── create_dataset.py           # Script to generate synthetic training data
│   └── disease_symptom_data.csv    # Generated symptom-disease dataset
├── models/
│   ├── disease_prediction_model.pkl  # Trained ML classifier
│   ├── disease_list.pkl              # List of all disease classes
│   ├── feature_importance.pkl        # Feature importance scores
│   └── symptom_names.pkl             # Ordered list of symptom features
├── notebook/
│   └── disease_prediction_model.ipynb  # Model training & EDA notebook
├── README.md
└── requirements.txt
```

---

##  Features

- **Multi-role login** — Doctor, Patient, Nurse, Admin, and Data Analyst views
- **Disease prediction** — Top 3 predicted diseases with confidence scores
- **Symptom importance chart** — Visual breakdown of which symptoms drove the prediction
- **Analytics dashboard** — Prediction logs, disease frequency, role distribution
- **Patient history** — Per-patient prediction timeline and confidence trends
- **Admin dashboard** — Overview of all patient histories and exportable CSV logs

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/disease_prediction.git
cd disease_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset (optional — CSV is already included)

```bash
python data/create_dataset.py
```

### 4. Train the model (optional — `.pkl` files are already included)

Open and run `notebook/disease_prediction_model.ipynb` end-to-end.

### 5. Run the app

```bash
streamlit run app.py
```

---

##  Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Training samples | 1,200 (100 per disease) |
| Number of diseases | 12 |
| Number of symptoms | 37 |
| Input format | Binary symptom vector |

**Diseases covered:** Common Cold, Flu, COVID-19, Migraine, Diabetes, Hypertension, Asthma, Pneumonia, Gastritis, UTI, Allergies, Bronchitis

---

##  Tech Stack

- **Frontend** — Streamlit
- **ML** — Scikit-learn, NumPy, Pandas
- **Visualization** — Plotly Express
- **Model persistence** — Joblib

---

##  Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

> Install all with: `pip install -r requirements.txt`

