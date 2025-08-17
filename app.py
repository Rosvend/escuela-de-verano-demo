# CardioML Medellín: Cardiovascular Risk Prediction Demo
# This single script generates synthetic data, trains a model, and runs a Streamlit web app.
#
# HOW TO RUN THIS DEMO:
# 1. Make sure you have Python installed.
# 2. Install the necessary libraries by opening your terminal or command prompt and running:
#    pip install pandas numpy scikit-learn xgboost streamlit joblib
# 3. Save this code as a Python file (e.g., app.py).
# 4. In your terminal, navigate to the folder where you saved the file.
# 5. Run the following command:
#    streamlit run app.py
# 6. Your web browser will open with the interactive demo.

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- PART 1: SYNTHETIC DATA GENERATION ---
# We'll create a function to generate data that looks like it could come from Medellín's older population.
def generate_synthetic_data(num_records=5000):
    """
    Generates a Pandas DataFrame with synthetic patient data plausible for Medellín.
    The data includes correlations between risk factors and the target variable.
    """
    np.random.seed(42) # for reproducibility

    # Define plausible distributions and choices for Medellín
    data = {
        'Edad': np.random.normal(loc=68, scale=5, size=num_records).astype(int),
        'Sexo': np.random.choice(['Femenino', 'Masculino'], size=num_records, p=[0.55, 0.45]),
        'Estrato_Socioeconomico': np.random.choice([1, 2, 3, 4, 5, 6], size=num_records, p=[0.13, 0.33, 0.31, 0.11, 0.08, 0.04]),
        'Comuna': np.random.randint(1, 17, size=num_records),
        'Presion_Arterial_Sistolica': np.random.normal(loc=135, scale=15, size=num_records).astype(int),
        'Colesterol_Total': np.random.normal(loc=200, scale=30, size=num_records).astype(int),
        'IMC': np.random.normal(loc=28, scale=4, size=num_records),
        'Tabaquismo': np.random.choice(['No', 'Sí'], size=num_records, p=[0.82, 0.18]),
        'Diabetes': np.random.choice(['No', 'Sí'], size=num_records, p=[0.85, 0.15]),
        'Antecedentes_Familiares': np.random.choice(['No', 'Sí'], size=num_records, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)

    # Clip values to be within a realistic range
    df['Edad'] = df['Edad'].clip(60, 85)
    df['Presion_Arterial_Sistolica'] = df['Presion_Arterial_Sistolica'].clip(90, 200)
    df['Colesterol_Total'] = df['Colesterol_Total'].clip(130, 300)
    df['IMC'] = df['IMC'].clip(18, 45)

    # --- Create the Target Variable (Evento_Cardiovascular) ---
    # We'll create a "risk score" based on the features to ensure the model can learn.
    # Higher values for these features will increase the probability of a cardiovascular event.
    
    # Normalize features to a 0-1 scale for scoring
    sistolica_norm = (df['Presion_Arterial_Sistolica'] - 90) / (200 - 90)
    edad_norm = (df['Edad'] - 60) / (85 - 60)
    colesterol_norm = (df['Colesterol_Total'] - 130) / (300 - 130)
    imc_norm = (df['IMC'] - 18) / (45 - 18)

    # Calculate risk score with some noise
    risk_score = (
        0.35 * edad_norm +
        0.25 * sistolica_norm +
        0.15 * imc_norm +
        0.10 * colesterol_norm +
        0.05 * (df['Tabaquismo'] == 'Sí') +
        0.05 * (df['Diabetes'] == 'Sí') +
        0.05 * (df['Antecedentes_Familiares'] == 'Sí') +
        np.random.normal(0, 0.08, size=num_records) # Add randomness
    )

    # Create the binary outcome based on the risk score
    # We set a threshold (e.g., the 80th percentile) to create an imbalanced dataset, which is more realistic.
    event_threshold = risk_score.quantile(0.80)
    df['Evento_Cardiovascular'] = (risk_score > event_threshold).astype(int)

    return df

# --- PART 2: MODEL TRAINING ---
def train_model(df):
    """
    Trains an XGBoost classifier on the synthetic data and saves it to a file.
    """
    # Preprocessing: Convert categorical variables to numerical format
    categorical_cols = ['Sexo', 'Tabaquismo', 'Diabetes', 'Antecedentes_Familiares']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Store the mapping for later use in the app
        le_path = f'{col}_encoder.joblib'
        joblib.dump(le, le_path)


    # Define features (X) and target (y)
    features = [
        'Edad', 'Sexo', 'Estrato_Socioeconomico', 'Presion_Arterial_Sistolica',
        'Colesterol_Total', 'IMC', 'Tabaquismo', 'Diabetes', 'Antecedentes_Familiares'
    ]
    X = df[features]
    y = df['Evento_Cardiovascular']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the XGBoost model
    # The parameters are set to handle the imbalanced dataset
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight= (y_train == 0).sum() / (y_train == 1).sum(), # Handle class imbalance
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'xgboost_cardio_model.joblib')
    print("Model trained and saved successfully.")

# --- Main script logic ---
# This part ensures that the data generation and model training only run once.
MODEL_FILE = 'xgboost_cardio_model.joblib'

if not os.path.exists(MODEL_FILE):
    print("Model file not found. Generating data and training a new model...")
    synthetic_data = generate_synthetic_data()
    train_model(synthetic_data)
else:
    print("Model file found. Loading existing model.")

# Load the pre-trained model
model = joblib.load(MODEL_FILE)

# Load the encoders
encoders = {}
for col in ['Sexo', 'Tabaquismo', 'Diabetes', 'Antecedentes_Familiares']:
    try:
        encoders[col] = joblib.load(f'{col}_encoder.joblib')
    except FileNotFoundError:
        st.error(f"Error: Encoder for {col} not found. Please retrain the model.")
        st.stop()


# --- PART 3: STREAMLIT WEB APPLICATION ---
st.set_page_config(page_title="CardioML Medellín", page_icon="❤️", layout="centered")

# --- UI Elements ---
st.title("❤️ CardioML Medellín")
st.subheader("Prototipo para Estimación de Riesgo Cardiovascular en Población Mayor")
st.markdown("""
Esta es una **demostración** para ilustrar el objetivo de nuestro proyecto. 
Utiliza un modelo de IA entrenado con **datos sintéticos** que simulan las características 
de la población de Medellín. **No debe ser utilizado para decisiones médicas reales.**
""")
st.markdown("---")

# --- Input Form in the Sidebar ---
st.sidebar.header("Ingrese los Datos del Paciente Ficticio")

def user_input_features():
    """
    Creates sidebar widgets to collect user input.
    """
    edad = st.sidebar.slider('Edad', 60, 85, 68)
    sexo = st.sidebar.radio('Sexo', ('Femenino', 'Masculino'))
    estrato = st.sidebar.selectbox('Estrato Socioeconómico', [1, 2, 3, 4, 5, 6], index=2)
    sistolica = st.sidebar.slider('Presión Arterial Sistólica (mmHg)', 90, 200, 135)
    colesterol = st.sidebar.slider('Colesterol Total (mg/dL)', 130, 300, 200)
    imc = st.sidebar.slider('Índice de Masa Corporal (IMC)', 18.0, 45.0, 28.0, 0.1)
    tabaquismo = st.sidebar.radio('¿Fuma actualmente?', ('No', 'Sí'))
    diabetes = st.sidebar.radio('¿Tiene diagnóstico de Diabetes?', ('No', 'Sí'))
    antecedentes = st.sidebar.radio('¿Antecedentes familiares de infarto o ACV?', ('No', 'Sí'))

    data = {
        'Edad': edad,
        'Sexo': sexo,
        'Estrato_Socioeconomico': estrato,
        'Presion_Arterial_Sistolica': sistolica,
        'Colesterol_Total': colesterol,
        'IMC': imc,
        'Tabaquismo': tabaquismo,
        'Diabetes': diabetes,
        'Antecedentes_Familiares': antecedentes
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display User Input ---
st.subheader("Datos Ingresados por el Usuario:")
st.write(input_df)

# --- Prediction Logic ---
if st.button('Calcular Riesgo', key='predict_button'):
    # Preprocess the input data to match the model's training format
    predict_df = input_df.copy()
    for col, encoder in encoders.items():
        # Use the fitted encoder to transform the input
        predict_df[col] = encoder.transform(predict_df[col])
    
    # Reorder columns to match the training order
    feature_order = [
        'Edad', 'Sexo', 'Estrato_Socioeconomico', 'Presion_Arterial_Sistolica',
        'Colesterol_Total', 'IMC', 'Tabaquismo', 'Diabetes', 'Antecedentes_Familiares'
    ]
    predict_df = predict_df[feature_order]

    # Get the prediction probability
    prediction_proba = model.predict_proba(predict_df)[:, 1][0] # Probability of class '1' (event)

    st.markdown("---")
    st.subheader('Resultado de la Predicción')

    # Display the result using a metric card
    risk_percentage = prediction_proba * 100
    st.metric(
        label="Riesgo de Evento Cardiovascular en los Próximos 10 Años",
        value=f"{risk_percentage:.1f}%"
    )

    # Provide a qualitative interpretation based on the risk
    if risk_percentage < 10:
        st.success("🟢 **Riesgo Bajo:** La probabilidad estimada es relativamente baja. Se recomienda mantener un estilo de vida saludable.")
    elif risk_percentage < 25:
        st.warning("🟡 **Riesgo Intermedio:** Se ha detectado una probabilidad moderada. Es un buen momento para discutir estrategias de prevención con un profesional de la salud.")
    else:
        st.error("🔴 **Riesgo Alto:** La probabilidad estimada es elevada. Se recomienda una consulta médica para una evaluación completa y un plan de manejo.")

    st.info("**Nota Importante:** Este resultado es una estimación generada por un modelo de demostración y no reemplaza el juicio clínico de un profesional de la salud.", icon="ℹ️")

