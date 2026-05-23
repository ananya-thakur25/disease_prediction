import pandas as pd
import numpy as np

# Define diseases and their associated symptoms
disease_symptoms = {
    'Common Cold': ['runny_nose', 'sneezing', 'sore_throat', 'cough', 'mild_fever', 'fatigue'],
    'Flu': ['high_fever', 'body_ache', 'fatigue', 'headache', 'cough', 'chills'],
    'COVID-19': ['high_fever', 'dry_cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath'],
    'Migraine': ['severe_headache', 'nausea', 'sensitivity_to_light', 'vision_changes', 'dizziness'],
    'Diabetes': ['increased_thirst', 'frequent_urination', 'fatigue', 'blurred_vision', 'slow_healing'],
    'Hypertension': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath', 'nosebleed'],
    'Asthma': ['shortness_of_breath', 'wheezing', 'chest_tightness', 'cough', 'difficulty_breathing'],
    'Pneumonia': ['high_fever', 'cough', 'chest_pain', 'shortness_of_breath', 'fatigue', 'chills'],
    'Gastritis': ['stomach_pain', 'nausea', 'vomiting', 'loss_of_appetite', 'bloating'],
    'UTI': ['frequent_urination', 'burning_sensation', 'lower_abdominal_pain', 'cloudy_urine', 'fever'],
    'Allergies': ['sneezing', 'runny_nose', 'itchy_eyes', 'rash', 'congestion'],
    'Bronchitis': ['cough', 'mucus_production', 'chest_discomfort', 'fatigue', 'mild_fever'],
}

# All unique symptoms
all_symptoms = sorted(list(set([symptom for symptoms in disease_symptoms.values() for symptom in symptoms])))

# Generate dataset
data = []
for disease, primary_symptoms in disease_symptoms.items():
    # Generate 100 samples per disease
    for _ in range(100):
        row = {symptom: 0 for symptom in all_symptoms}
        
        # Select 3-6 random symptoms from primary symptoms
        num_symptoms = np.random.randint(3, min(7, len(primary_symptoms) + 1))
        selected_symptoms = np.random.choice(primary_symptoms, size=num_symptoms, replace=False)
        
        for symptom in selected_symptoms:
            row[symptom] = 1
        
        # Add some noise (occasional unrelated symptoms)
        if np.random.random() < 0.2:
            noise_symptom = np.random.choice([s for s in all_symptoms if s not in primary_symptoms])
            row[noise_symptom] = 1
        
        row['disease'] = disease
        data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('data/disease_symptom_data.csv', index=False)

print(f"Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nDiseases: {df['disease'].unique()}")
print(f"Number of symptoms: {len(all_symptoms)}")
print(f"\nFirst few rows:")
print(df.head())