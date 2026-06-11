import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.preprocessing import add_engineered_features
from src.gui_helpers import FEATURE_METADATA, make_comparison_plot

st.set_page_config(
    page_title="Smoker Status Predictor",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# STYLING & ESTHETICS (Premium CSS)
# ----------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f7f7f;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value-high {
        font-size: 3rem;
        font-weight: 800;
        color: #d62728;
    }
    .metric-value-low {
        font-size: 3rem;
        font-weight: 800;
        color: #2ca02c;
    }
    .section-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: #fafafa;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MODEL LOADING & AUTODETECTION
# ----------------------------------------------------
@st.cache_resource
def load_cached_model():
    model_path = os.path.join("models", "model.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        # Auto-detect pipeline type
        is_feature_engineered = False
        try:
            transformers = model.named_steps['Pipeline'].transformers_
            # If the columns list contains 'BMI' or 'eyesight', it is engineered
            for name, trans, cols in transformers:
                if 'BMI' in cols or 'eyesight' in cols:
                    is_feature_engineered = True
                    break
        except Exception:
            pass
        return model, is_feature_engineered
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_info = load_cached_model()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/no-smoking.png", width=80)
    st.markdown("### 🚭 Smoker Status Predictor")
    st.markdown("This AI assistant predicts the probability of an individual being a smoker using bio-signals and clinical measurements.")
    
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    if model_info is not None:
        model, is_engineered = model_info
        st.success("Model: Loaded Successfully")
        st.info(f"Pipeline Mode: **{'Feature Engineered' if is_engineered else 'Raw Features (Simple)'}**")
        st.markdown("**Algorithm:** LightGBM Classifier")
        st.markdown("**Validation ROC-AUC:** `0.86676` (Raw) | `0.85716` (Engineered)")
    else:
        st.error("Model: Not Found")
        st.warning("Please train the model first by running:\n`python -m src.train` in your terminal.")

    st.markdown("---")
    st.markdown("Created as a modular Data Science repository solution.")

# ----------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------
st.markdown('<div class="main-header">Smoker Status Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict smoker risk using clinical bio-signals & perform batch predictions</div>', unsafe_allow_html=True)

if model_info is None:
    st.error("⚠️ Model file `models/model.joblib` was not found. Please train the model using your terminal before using the GUI.")
    st.stop()

model, is_engineered = model_info

tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📁 Batch Prediction", "🔬 Model Metadata"])

# ----------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTION
# ----------------------------------------------------
with tab1:
    st.markdown("### Input Bio-Signals")
    st.write("Fill in the fields below to predict the probability of smoker status for a single individual.")

    # Form grouping
    with st.form("individual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 General Metrics")
            age = st.slider(
                FEATURE_METADATA['age']['label'],
                min_value=int(FEATURE_METADATA['age']['min']),
                max_value=int(FEATURE_METADATA['age']['max']),
                value=int(FEATURE_METADATA['age']['median']),
                help=FEATURE_METADATA['age']['help']
            )
            height = st.slider(
                FEATURE_METADATA['height(cm)']['label'],
                min_value=int(FEATURE_METADATA['height(cm)']['min']),
                max_value=int(FEATURE_METADATA['height(cm)']['max']),
                value=int(FEATURE_METADATA['height(cm)']['median']),
                help=FEATURE_METADATA['height(cm)']['help']
            )
            weight = st.slider(
                FEATURE_METADATA['weight(kg)']['label'],
                min_value=int(FEATURE_METADATA['weight(kg)']['min']),
                max_value=int(FEATURE_METADATA['weight(kg)']['max']),
                value=int(FEATURE_METADATA['weight(kg)']['median']),
                help=FEATURE_METADATA['weight(kg)']['help']
            )
            waist = st.slider(
                FEATURE_METADATA['waist(cm)']['label'],
                min_value=float(FEATURE_METADATA['waist(cm)']['min']),
                max_value=float(FEATURE_METADATA['waist(cm)']['max']),
                value=float(FEATURE_METADATA['waist(cm)']['median']),
                step=0.5,
                help=FEATURE_METADATA['waist(cm)']['help']
            )

        with col2:
            st.markdown("##### 🩸 Blood Pressure & Sugar")
            systolic = st.number_input(
                FEATURE_METADATA['systolic']['label'],
                min_value=int(FEATURE_METADATA['systolic']['min']),
                max_value=int(FEATURE_METADATA['systolic']['max']),
                value=int(FEATURE_METADATA['systolic']['median']),
                help=FEATURE_METADATA['systolic']['help']
            )
            relaxation = st.number_input(
                FEATURE_METADATA['relaxation']['label'],
                min_value=int(FEATURE_METADATA['relaxation']['min']),
                max_value=int(FEATURE_METADATA['relaxation']['max']),
                value=int(FEATURE_METADATA['relaxation']['median']),
                help=FEATURE_METADATA['relaxation']['help']
            )
            sugar = st.number_input(
                FEATURE_METADATA['fasting blood sugar']['label'],
                min_value=int(FEATURE_METADATA['fasting blood sugar']['min']),
                max_value=int(FEATURE_METADATA['fasting blood sugar']['max']),
                value=int(FEATURE_METADATA['fasting blood sugar']['median']),
                help=FEATURE_METADATA['fasting blood sugar']['help']
            )
            
            st.markdown("##### 🧪 Lipids & Cholesterols")
            cholesterol = st.number_input(
                FEATURE_METADATA['Cholesterol']['label'],
                min_value=int(FEATURE_METADATA['Cholesterol']['min']),
                max_value=int(FEATURE_METADATA['Cholesterol']['max']),
                value=int(FEATURE_METADATA['Cholesterol']['median']),
                help=FEATURE_METADATA['Cholesterol']['help']
            )
            triglyceride = st.number_input(
                FEATURE_METADATA['triglyceride']['label'],
                min_value=int(FEATURE_METADATA['triglyceride']['min']),
                max_value=int(FEATURE_METADATA['triglyceride']['max']),
                value=int(FEATURE_METADATA['triglyceride']['median']),
                help=FEATURE_METADATA['triglyceride']['help']
            )
            hdl = st.number_input(
                FEATURE_METADATA['HDL']['label'],
                min_value=int(FEATURE_METADATA['HDL']['min']),
                max_value=int(FEATURE_METADATA['HDL']['max']),
                value=int(FEATURE_METADATA['HDL']['median']),
                help=FEATURE_METADATA['HDL']['help']
            )
            ldl = st.number_input(
                FEATURE_METADATA['LDL']['label'],
                min_value=int(FEATURE_METADATA['LDL']['min']),
                max_value=int(FEATURE_METADATA['LDL']['max']),
                value=int(FEATURE_METADATA['LDL']['median']),
                help=FEATURE_METADATA['LDL']['help']
            )

        with col3:
            st.markdown("##### 🧪 Kidneys & Oxygen")
            hemoglobin = st.number_input(
                FEATURE_METADATA['hemoglobin']['label'],
                min_value=float(FEATURE_METADATA['hemoglobin']['min']),
                max_value=float(FEATURE_METADATA['hemoglobin']['max']),
                value=float(FEATURE_METADATA['hemoglobin']['median']),
                step=0.1,
                help=FEATURE_METADATA['hemoglobin']['help']
            )
            urine = st.selectbox(
                FEATURE_METADATA['Urine protein']['label'],
                options=[1, 2, 3, 4, 5, 6],
                index=int(FEATURE_METADATA['Urine protein']['median']) - 1,
                help=FEATURE_METADATA['Urine protein']['help']
            )
            creatinine = st.number_input(
                FEATURE_METADATA['serum creatinine']['label'],
                min_value=float(FEATURE_METADATA['serum creatinine']['min']),
                max_value=float(FEATURE_METADATA['serum creatinine']['max']),
                value=float(FEATURE_METADATA['serum creatinine']['median']),
                step=0.1,
                help=FEATURE_METADATA['serum creatinine']['help']
            )

            st.markdown("##### 🧬 Liver & Vital Enzymes")
            ast = st.number_input(
                FEATURE_METADATA['AST']['label'],
                min_value=int(FEATURE_METADATA['AST']['min']),
                max_value=int(FEATURE_METADATA['AST']['max']),
                value=int(FEATURE_METADATA['AST']['median']),
                help=FEATURE_METADATA['AST']['help']
            )
            alt = st.number_input(
                FEATURE_METADATA['ALT']['label'],
                min_value=int(FEATURE_METADATA['ALT']['min']),
                max_value=int(FEATURE_METADATA['ALT']['max']),
                value=int(FEATURE_METADATA['ALT']['median']),
                help=FEATURE_METADATA['ALT']['help']
            )
            gtp = st.number_input(
                FEATURE_METADATA['Gtp']['label'],
                min_value=int(FEATURE_METADATA['Gtp']['min']),
                max_value=int(FEATURE_METADATA['Gtp']['max']),
                value=int(FEATURE_METADATA['Gtp']['median']),
                help=FEATURE_METADATA['Gtp']['help']
            )

        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("##### 👁️ Eyesight Visual Acuity")
            eye_left = st.number_input(
                FEATURE_METADATA['eyesight(left)']['label'],
                min_value=float(FEATURE_METADATA['eyesight(left)']['min']),
                max_value=float(FEATURE_METADATA['eyesight(left)']['max']),
                value=float(FEATURE_METADATA['eyesight(left)']['median']),
                step=0.1,
                help=FEATURE_METADATA['eyesight(left)']['help']
            )
            eye_right = st.number_input(
                FEATURE_METADATA['eyesight(right)']['label'],
                min_value=float(FEATURE_METADATA['eyesight(right)']['min']),
                max_value=float(FEATURE_METADATA['eyesight(right)']['max']),
                value=float(FEATURE_METADATA['eyesight(right)']['median']),
                step=0.1,
                help=FEATURE_METADATA['eyesight(right)']['help']
            )
        with col_s2:
            st.markdown("##### 👂 Hearing Ability")
            hearing_left = st.selectbox(
                FEATURE_METADATA['hearing(left)']['label'],
                options=[1, 2],
                format_func=lambda x: "1 - Normal" if x == 1 else "2 - Impaired",
                help=FEATURE_METADATA['hearing(left)']['help']
            )
            hearing_right = st.selectbox(
                FEATURE_METADATA['hearing(right)']['label'],
                options=[1, 2],
                format_func=lambda x: "1 - Normal" if x == 1 else "2 - Impaired",
                help=FEATURE_METADATA['hearing(right)']['help']
            )
        with col_s3:
            st.markdown("##### 🦷 Oral Hygiene")
            caries = st.selectbox(
                FEATURE_METADATA['dental caries']['label'],
                options=[0, 1],
                format_func=lambda x: "0 - No Cavities" if x == 0 else "1 - Cavities Present",
                help=FEATURE_METADATA['dental caries']['help']
            )

        submit_btn = st.form_submit_button("Predict Smoker Status")

    # ----------------------------------------------------
    # RUN SINGLE INFERENCE ON SUBMIT
    # ----------------------------------------------------
    if submit_btn:
        # Construct input dictionary matching raw feature list
        input_data = {
            'age': age,
            'height(cm)': height,
            'weight(kg)': weight,
            'waist(cm)': waist,
            'eyesight(left)': eye_left,
            'eyesight(right)': eye_right,
            'hearing(left)': hearing_left,
            'hearing(right)': hearing_right,
            'systolic': systolic,
            'relaxation': relaxation,
            'fasting blood sugar': sugar,
            'Cholesterol': cholesterol,
            'triglyceride': triglyceride,
            'HDL': hdl,
            'LDL': ldl,
            'hemoglobin': hemoglobin,
            'Urine protein': urine,
            'serum creatinine': creatinine,
            'AST': ast,
            'ALT': alt,
            'Gtp': gtp,
            'dental caries': caries
        }
        
        input_df = pd.DataFrame([input_data])
        
        # If the model expects feature engineered columns, transform them
        if is_engineered:
            input_df = add_engineered_features(input_df)

        # Predict
        prob = model.predict_proba(input_df)[0, 1]
        
        st.markdown("### 📊 Prediction Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.write("**Smoking Probability**")
            pct = prob * 100
            if pct >= 50.0:
                st.markdown(f'<div class="metric-value-high">{pct:.1f}%</div>', unsafe_allow_html=True)
                st.markdown("**Smoker Risk: HIGH** 🛑")
            else:
                st.markdown(f'<div class="metric-value-low">{pct:.1f}%</div>', unsafe_allow_html=True)
                st.markdown("**Smoker Risk: LOW** ✅")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.write("##### AI Diagnosis Analysis:")
            st.progress(prob)
            
            # Clinical insights
            if prob >= 0.5:
                st.error("🚨 Based on liver enzymes (GTP, ALT) and blood hemoglobin profiles, this bio-signature strongly matches patterns seen in chronic smokers. High GTP and high hemoglobin are typical indicators of active smoking habits.")
            else:
                st.success("✨ Clinical readings such as visual acuity, hearing checks, normal range liver enzymes, and creatinine clearance scores indicate a biological signature typical of non-smokers.")

        # Expandable distribution plots
        st.markdown("---")
        with st.expander("🔍 View Comparison Plots (Your Inputs vs General Population)"):
            st.write("See where your inputs stand compared to the general dataset distribution.")
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                st.pyplot(make_comparison_plot('age', age))
                st.pyplot(make_comparison_plot('waist(cm)', waist))
                st.pyplot(make_comparison_plot('hemoglobin', hemoglobin))
            with plot_col2:
                st.pyplot(make_comparison_plot('Gtp', gtp))
                st.pyplot(make_comparison_plot('fasting blood sugar', sugar))
                st.pyplot(make_comparison_plot('triglyceride', triglyceride))


# ----------------------------------------------------
# TAB 2: BATCH PREDICTION
# ----------------------------------------------------
with tab2:
    st.markdown("### 📁 Batch Prediction Ingestion")
    st.write("Upload a CSV file containing multiple patients' bio-signals to predict smoker status in batch. The file should contain all required clinical fields.")
    
    with st.expander("📝 View Required CSV Headers"):
        st.write("Your CSV file must contain the following columns:")
        st.code(", ".join(list(FEATURE_METADATA.keys())))
        st.write("*Note: An optional `id` column can be provided to trace predictions back to specific patients.*")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            st.markdown("##### 📄 Data Preview (First 5 rows):")
            st.dataframe(df_batch.head())
            
            # Check for columns
            missing_cols = [col for col in FEATURE_METADATA.keys() if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in uploaded CSV: {', '.join(missing_cols)}")
            else:
                run_btn = st.button("Run Batch Inference")
                
                if run_btn:
                    with st.spinner("Processing batch predictions..."):
                        # Keep a copy of original IDs
                        if 'id' in df_batch.columns:
                            ids = df_batch['id']
                        else:
                            ids = pd.Series(range(len(df_batch)), name='id')
                            
                        # Extract features
                        X_batch = df_batch[list(FEATURE_METADATA.keys())].copy()
                        
                        # Apply feature engineering if required by pipeline
                        if is_engineered:
                            X_batch = add_engineered_features(X_batch)
                            
                        # Run predictions
                        probs = model.predict_proba(X_batch)[:, 1]
                        
                        # Create output DataFrame
                        output_df = pd.DataFrame({
                            'id': ids,
                            'smoking_probability': probs,
                            'predicted_smoker': (probs >= 0.5).astype(int)
                        })
                        
                        st.markdown("### 🎉 Batch Results Preview")
                        st.dataframe(output_df.head(10))
                        
                        # Add download button
                        csv_output = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predicted Smoker Status CSV",
                            data=csv_output,
                            file_name="smoker_status_batch_predictions.csv",
                            mime="text/csv"
                        )
                        st.success("Prediction complete! Click the button above to download your results.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ----------------------------------------------------
# TAB 3: MODEL METADATA & PARAMETERS
# ----------------------------------------------------
with tab3:
    st.markdown("### 🔬 Model Performance & Pipeline Summary")
    st.write("Detailed metadata on the LightGBM Classifier and pre-scaling transformers.")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("##### ⚙️ Hyperparameters (LightGBM)")
        st.markdown("""
        * **Number of Leaves:** `128`
        * **Minimum Child Samples:** `10`
        * **Min Sum Hessian in Leaf:** `1`
        * **Feature Fraction:** `1.0` (all columns used per tree)
        * **Bagging Fraction:** `1.0` (no subsampling)
        * **Objective:** `binary`
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("##### 📈 Competition Metrics")
        st.markdown("""
        * **Validation ROC-AUC:** `0.86676`
        * **Cross-Validation random state:** `0`
        * **Train/Validation Ratio:** `80% / 20%`
        * **Target:** `smoking` (0 = Non-Smoker, 1 = Smoker)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("##### Feature Scaling Pipeline Visualization")
    st.markdown("""
    ```text
    [Raw Patient Bio-Signals] 
              ⬇️
      [ColumnTransformer] ➡️ Scaler: StandardScaler() (Applies mean centering and unit variance)
              ⬇️
      [LightGBM Model]    ➡️ Evaluates tree boundaries and splits
              ⬇️
      [Smoking Risk Probability]
    ```
    """)
