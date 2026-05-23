import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import os

# ============================================================
# Streamlit Page Configuration
# ============================================================
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom Styling
# ============================================================
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #45a049; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content { background-color: #f5f7fa; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
for key, default in {
    'prediction_logs': [],
    'patient_history': {},
    'current_user': None,
    'user_role': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# Safe Model Loader
# ============================================================
@st.cache_resource
def load_model():
    """Safely load ML model and related assets."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_path, "models")

    try:
        model = joblib.load(os.path.join(model_dir, "disease_prediction_model.pkl"))
        symptom_names = joblib.load(os.path.join(model_dir, "symptom_names.pkl"))
        feature_importance = joblib.load(os.path.join(model_dir, "feature_importance.pkl"))
        disease_list = joblib.load(os.path.join(model_dir, "disease_list.pkl"))
        return model, symptom_names, feature_importance, disease_list
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None, None

model, symptom_names, feature_importance, disease_list = load_model()

# ============================================================
# Utility Functions
# ============================================================
def predict_disease(symptoms_input, model, symptom_names):
    """Predict disease based on selected symptoms."""
    start_time = datetime.now()
    features = np.zeros(len(symptom_names))
    for symptom in symptoms_input:
        if symptom in symptom_names:
            features[symptom_names.index(symptom)] = 1
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = model.classes_[top_3_idx]
    top_3_probs = probabilities[top_3_idx]
    results = [(disease, prob) for disease, prob in zip(top_3_diseases, top_3_probs)]
    response_time = (datetime.now() - start_time).total_seconds()
    return results, response_time


def get_symptom_importance(symptoms, feature_importance_df):
    """Return feature importance for selected symptoms."""
    importance_data = []
    for symptom in symptoms:
        if symptom in feature_importance_df['symptom'].values:
            imp = feature_importance_df.loc[
                feature_importance_df['symptom'] == symptom, 'importance'
            ].values[0]
            importance_data.append({'symptom': symptom, 'importance': imp})
    return pd.DataFrame(importance_data).sort_values('importance', ascending=False)


def export_logs_to_csv():
    """Allow admin to export prediction logs as CSV."""
    if not st.session_state.prediction_logs:
        st.warning("No logs available for export.")
        return
    df = pd.DataFrame(st.session_state.prediction_logs)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Logs as CSV", data=csv, file_name="prediction_logs.csv", mime="text/csv")


# ============================================================
# Login Page
# ============================================================
def show_login():
    st.title("🏥 Disease Prediction System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Login")
        username = st.text_input("Username", placeholder="Enter your username")
        role = st.selectbox("Select Role", ["Doctor", "Patient", "Nurse", "Admin", "Data Analyst"])

        if st.button("Login"):
            if username:
                st.session_state.current_user = username
                st.session_state.user_role = role
                st.rerun()
            else:
                st.error("Please enter a username.")

# ============================================================
# Home Page
# ============================================================
def show_home_page():
    st.title("🏥 Disease Prediction System")
    st.markdown("### Welcome to the AI-powered Medical Diagnosis Assistant")

    cols = st.columns(3)
    features = [
        ("🎯", "Accurate", "ML-powered predictions"),
        ("⚡", "Fast", "Get results in seconds"),
        ("🔒", "Secure", "Your data is private and safe")
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{icon}</h2>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# Prediction Page
# ============================================================
def show_prediction_page():
    st.title("🔍 Disease Prediction")

    if model is None:
        st.error("Model not loaded. Ensure files exist under `/models` directory.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Select Patient Symptoms")
        symptom_display = [s.replace('_', ' ').title() for s in symptom_names]
        symptom_dict = dict(zip(symptom_display, symptom_names))
        selected_symptoms_display = st.multiselect(
            "Search and select symptoms:",
            options=symptom_display,
            help="Select relevant symptoms"
        )
        selected_symptoms = [symptom_dict[s] for s in selected_symptoms_display]

        if selected_symptoms:
            st.info(f"✓ {len(selected_symptoms)} symptoms selected")

        predict_button = st.button("🔮 Predict Disease", type="primary")

    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Diseases", len(disease_list) if disease_list is not None else 0)
        st.metric("Symptoms in Database", len(symptom_names) if symptom_names is not None else 0)
        st.metric("Predictions Today", len(st.session_state.prediction_logs))

    if predict_button:
        if not selected_symptoms:
            st.error("⚠️ Please select at least one symptom.")
        elif len(selected_symptoms) < 2:
            st.warning("⚠️ Select at least 2 symptoms for better accuracy.")
        else:
            with st.spinner("🔄 Analyzing symptoms..."):
                results, response_time = predict_disease(selected_symptoms, model, symptom_names)
                log_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'user': st.session_state.current_user,
                    'role': st.session_state.user_role,
                    'symptoms': selected_symptoms,
                    'predictions': results,
                    'response_time': response_time,
                    'accuracy': results[0][1]
                }
                st.session_state.prediction_logs.append(log_entry)

                if st.session_state.user_role == "Patient":
                    user = st.session_state.current_user
                    st.session_state.patient_history.setdefault(user, []).append(log_entry)

                st.success(f"✅ Prediction completed in {response_time:.3f}s")

                st.markdown("---")
                st.subheader("🎯 Top 3 Disease Predictions")
                for i, (disease, prob) in enumerate(results, 1):
                    st.write(f"**{i}. {disease}** — Confidence: {prob:.1%}")
                    st.progress(prob)

                st.markdown("---")
                st.subheader("📊 Symptom Importance")
                symptom_importance = get_symptom_importance(selected_symptoms, feature_importance)
                if not symptom_importance.empty:
                    fig = px.bar(
                        symptom_importance,
                        x='importance',
                        y='symptom',
                        orientation='h',
                        title='Contributing Symptoms',
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No importance data available for selected symptoms.")

# ============================================================
# Analytics Page
# ============================================================
def show_analytics_page():
    st.title("📈 Prediction Analytics")
    logs = st.session_state.prediction_logs
    if not logs:
        st.info("No analytics data available.")
        return

    logs_df = pd.DataFrame([
        {
            'timestamp': log['timestamp'],
            'user': log['user'],
            'role': log['role'],
            'top_disease': log['predictions'][0][0],
            'confidence': log['predictions'][0][1],
            'response_time': log['response_time'],
            'num_symptoms': len(log['symptoms'])
        }
        for log in logs
    ])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", len(logs_df))
    col2.metric("Avg Confidence", f"{logs_df['confidence'].mean():.1%}")
    col3.metric("Avg Response Time", f"{logs_df['response_time'].mean():.3f}s")
    col4.metric("Unique Users", logs_df['user'].nunique())

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.bar(
            logs_df['top_disease'].value_counts().head(10),
            orientation='h',
            title='Top 10 Predicted Diseases'
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.pie(
            logs_df,
            names='role',
            title='Predictions by Role'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📤 Export Logs")
    export_logs_to_csv()

# ============================================================
# Patient History Page
# ============================================================
def show_patient_history():
    st.title("🧾 Patient History")
    user = st.session_state.current_user
    if user not in st.session_state.patient_history:
        st.info("No past predictions found.")
        return

    history = st.session_state.patient_history[user]
    df = pd.DataFrame([
        {
            'timestamp': h['timestamp'],
            'top_disease': h['predictions'][0][0],
            'confidence': h['predictions'][0][1],
            'symptom_count': len(h['symptoms']),
            'response_time': h['response_time']
        } for h in history
    ])

    st.dataframe(df)
    fig = px.line(df, x='timestamp', y='confidence', title='Confidence Over Time')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Admin Page
# ============================================================
def show_admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    show_analytics_page()
    st.markdown("---")
    st.subheader("🗂 Patient Histories Overview")
    if not st.session_state.patient_history:
        st.info("No patient history available.")
        return
    for patient, logs in st.session_state.patient_history.items():
        st.markdown(f"**👤 {patient}** — {len(logs)} records")

# ============================================================
# Main App Navigation
# ============================================================
def main_app():
    with st.sidebar:
        st.title("🏥 Medical Dashboard")
        st.markdown(f"**User:** {st.session_state.current_user}")
        st.markdown(f"**Role:** {st.session_state.user_role}")
        st.markdown("---")

        role = st.session_state.user_role
        if role == "Doctor":
            menu = ["🏠 Home", "🔍 Prediction", "📊 Analytics"]
        elif role == "Patient":
            menu = ["🏠 Home", "🔍 Checker", "🧾 History"]
        elif role == "Admin":
            menu = ["📊 Analytics", "🛠️ Admin Dashboard"]
        else:
            menu = ["🏠 Home", "🔍 Prediction", "📊 Analytics"]

        selected_menu = st.radio("Navigation", menu)
        if st.button("Logout"):
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.rerun()

    if "Prediction" in selected_menu or "Checker" in selected_menu:
        show_prediction_page()
    elif "Analytics" in selected_menu:
        show_analytics_page()
    elif "History" in selected_menu:
        show_patient_history()
    elif "Admin" in selected_menu:
        show_admin_dashboard()
    else:
        show_home_page()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    if st.session_state.current_user is None:
        show_login()
    else:
        main_app()
