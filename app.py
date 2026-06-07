"""
老年神经重症患者肠内营养不耐受风险预警系统 V1.0
ENI Risk Prediction System for Elderly Neurocritical Care Patients
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import warnings
warnings.filterwarnings('ignore')

# Get base directory for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="ENI Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c5aa0;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 20px;
        color: #555;
        text-align: center;
        padding-bottom: 20px;
    }
    .risk-high {
        background-color: #ffcccc;
        color: #cc0000;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        background-color: #2c5aa0;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(BASE_DIR, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure best_model.pkl and scaler.pkl are in the working directory.")
        return None, None

model, scaler = load_model()

# ============================================================
# Feature Definitions (order must match model)
# ============================================================
SELECTED_FEATURES = [
    'Age', 'GCS', 'APACHE_II', 'SOFA', 'ICP', 'Surgery_History',
    'Hypothermia_Therapy', 'Albumin', 'Blood_Glucose', 'Mechanical_Ventilation',
    'Feeding_Route', 'EN_Start_Time', 'IAP', 'AGI_Grade', 'Sedation_Depth',
    'Vasoactive_Drugs', 'Diabetes_History'
]

# ============================================================
# Sidebar - Input Panel
# ============================================================
st.sidebar.markdown("## 📝 Patient Data Input")
st.sidebar.markdown("---")

# Demographics
st.sidebar.markdown("### Demographics")
age = st.sidebar.slider("Age (years)", 60, 90, 72)
gender = st.sidebar.selectbox("Gender", ["Female", " Male"], index=0)
bmi = st.sidebar.slider("BMI (kg/m²)", 15.0, 40.0, 23.0, 0.1)

# Severity Scores
st.sidebar.markdown("### Severity Scores")
gcs = st.sidebar.slider("GCS Score", 3, 15, 9)
apache_ii = st.sidebar.slider("APACHE II Score", 0, 40, 15)
sofa = st.sidebar.slider("SOFA Score", 0, 24, 5)
icp = st.sidebar.slider("ICP (mmHg)", 0.0, 30.0, 15.0, 0.1)

# Clinical Status
st.sidebar.markdown("### Clinical Status")
surgery_history = st.sidebar.selectbox("Surgery History", ["No", "Yes"], index=0)
hypothermia = st.sidebar.selectbox("Hypothermia Therapy", ["No", "Yes"], index=0)
mech_vent = st.sidebar.selectbox("Mechanical Ventilation", ["No", "Yes"], index=0)
vasoactive = st.sidebar.selectbox("Vasoactive Drugs", ["No", "Yes"], index=0)
sedation_depth = st.sidebar.selectbox("Sedation Depth", ["Mild", "Moderate", "Deep"], index=1)
diabetes = st.sidebar.selectbox("Diabetes History", ["No", "Yes"], index=0)

# Nutrition
st.sidebar.markdown("### Nutrition Parameters")
albumin = st.sidebar.slider("Albumin (g/L)", 20.0, 50.0, 32.0, 0.1)
blood_glucose = st.sidebar.slider("Blood Glucose (mmol/L)", 4.0, 20.0, 10.0, 0.1)
feeding_route = st.sidebar.selectbox("Feeding Route", ["Nasogastric", "Nasojejunal", "PEG"], index=0)
en_start_time = st.sidebar.slider("EN Start Time (hours)", 0.0, 72.0, 24.0, 0.5)
iap = st.sidebar.slider("IAP (mmHg)", 0.0, 25.0, 10.0, 0.1)
agi_grade = st.sidebar.selectbox("AGI Grade", ["Grade 0", "Grade I", "Grade II", "Grade III", "Grade IV"], index=0)

# ============================================================
# Convert inputs to model format
# ============================================================
def get_input_features():
    return np.array([[
        float(age),
        int(gcs),
        int(apache_ii),
        int(sofa),
        float(icp),
        1 if surgery_history == "Yes" else 0,
        1 if hypothermia == "Yes" else 0,
        float(albumin),
        float(blood_glucose),
        1 if mech_vent == "Yes" else 0,
        1 if feeding_route == "Nasogastric" else 2 if feeding_route == "Nasojejunal" else 3,
        float(en_start_time),
        float(iap),
        int(agi_grade.split()[-1]) if "Grade" in agi_grade else int(agi_grade),
        1 if sedation_depth == "Mild" else 2 if sedation_depth == "Moderate" else 3,
        1 if vasoactive == "Yes" else 0,
        1 if diabetes == "Yes" else 0
    ]])

# ============================================================
# Main Page
# ============================================================
st.markdown('<div class="main-header">🏥 ENI Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enteral Nutrition Intolerance Risk Prediction for Elderly Neurocritical Care Patients V1.0</div>', unsafe_allow_html=True)

# System Info
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-box">
        <h4>📊 Model</h4>
        <p>Logistic Regression with Lasso</p>
        <p>AUC: 0.924</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-box">
        <h4>🔬 Features</h4>
        <p>17 Clinical Variables</p>
        <p>Lasso-Selected from 25</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-box">
        <h4>✅ Validation</h4>
        <p>Bootstrap 1000×</p>
        <p>Corrected AUC: 0.925</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Prediction Button
# ============================================================
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔍 Predict ENI Risk", use_container_width=True)

if predict_clicked and model is not None:
    input_data = get_input_features()
    input_scaled = scaler.transform(input_data)
    risk_prob = model.predict_proba(input_scaled)[0, 1]
    risk_pct = risk_prob * 100
    
    # Risk stratification
    if risk_pct >= 70:
        risk_level = "High Risk"
        risk_class = "risk-high"
        advice = "⚠️ **High Risk**: Recommend early enteral nutrition monitoring, consider prokinetic agents, evaluate for post-pyloric feeding, and closely monitor gastric residual volume."
    elif risk_pct >= 40:
        risk_level = "Moderate Risk"
        risk_class = "risk-moderate"
        advice = "⚡ **Moderate Risk**: Monitor feeding tolerance every 4 hours, consider adjusting feeding rate, and evaluate for alternative feeding routes."
    else:
        risk_level = "Low Risk"
        risk_class = "risk-low"
        advice = "✅ **Low Risk**: Standard EN protocol is appropriate. Continue monitoring per routine care."
    
    # Display results
    st.markdown("---")
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        st.markdown(f"### Prediction Result")
        st.markdown(f'<div class="{risk_class}">{risk_level}<br>{risk_pct:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box" style="margin-top:15px">
            <h4>Clinical Recommendations</h4>
            <p>{advice}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature contribution (simplified SHAP-like)
        st.markdown("### 📋 Feature Contributions")
        coefficients = model.coef_[0]
        contributions = coefficients * input_scaled[0]
        contrib_df = pd.DataFrame({
            'Feature': SELECTED_FEATURES,
            'Value': input_data[0],
            'Contribution': contributions,
            'Direction': ['Increases Risk' if c > 0 else 'Decreases Risk' for c in contributions]
        }).sort_values('Contribution', key=abs, ascending=False)
        
        st.dataframe(contrib_df.style.apply(lambda row: ['background-color: #ffcccc' if row['Contribution'] > 0 else 'background-color: #d4edda'] * len(row), axis=1), use_container_width=True)
    
    with result_col2:
        # Gauge chart simulation
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_gauge = ['#2ca02c', '#ffdd57', '#d62728']
        ax.barh([0], [40], color=colors_gauge[0], height=0.5, label='Low Risk')
        ax.barh([0], [30], left=[40], color=colors_gauge[1], height=0.5, label='Moderate Risk')
        ax.barh([0], [30], left=[70], color=colors_gauge[2], height=0.5, label='High Risk')
        ax.scatter([risk_pct], [0], color='black', s=200, zorder=5, marker='|')
        ax.text(risk_pct, 0.35, f'{risk_pct:.1f}%', ha='center', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Risk Probability (%)', fontsize=12)
        ax.set_title('ENI Risk Gauge', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        st.pyplot(fig)
        
        # Key risk factors
        st.markdown("### 🔑 Key Risk Factors Identified")
        top_increasing = contrib_df[contrib_df['Contribution'] > 0].head(3)
        top_decreasing = contrib_df[contrib_df['Contribution'] < 0].head(2)
        
        for _, row in top_increasing.iterrows():
            st.markdown(f"- 🔴 **{row['Feature']}** ({row['Value']:.1f}): +{row['Contribution']:.3f}")
        for _, row in top_decreasing.iterrows():
            st.markdown(f"- 🟢 **{row['Feature']}** ({row['Value']:.1f}): {row['Contribution']:.3f}")
    
    # ============================================================
    # PDF Report Generation
    # ============================================================
    st.markdown("---")
    if st.button("📄 Generate PDF Report"):
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#2c5aa0'), spaceAfter=20, alignment=TA_CENTER)
            heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2c5aa0'), spaceAfter=10)
            normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=6)
            
            story = []
            
            # Title
            story.append(Paragraph("ENI Risk Prediction Report", title_style))
            story.append(Paragraph("Enteral Nutrition Intolerance Risk Prediction System V1.0", normal_style))
            story.append(Spacer(1, 20))
            
            # Risk Result
            story.append(Paragraph("Prediction Result", heading_style))
            risk_color = '#cc0000' if risk_pct >= 70 else '#856404' if risk_pct >= 40 else '#155724'
            story.append(Paragraph(f"<b>Risk Level:</b> <font color='{risk_color}'>{risk_level}</font>", normal_style))
            story.append(Paragraph(f"<b>Risk Probability:</b> {risk_pct:.2f}%", normal_style))
            story.append(Spacer(1, 15))
            
            # Clinical Recommendations
            story.append(Paragraph("Clinical Recommendations", heading_style))
            story.append(Paragraph(advice, normal_style))
            story.append(Spacer(1, 15))
            
            # Input Parameters
            story.append(Paragraph("Patient Parameters", heading_style))
            param_data = [['Parameter', 'Value']]
            param_labels = ['Age', 'GCS', 'APACHE II', 'SOFA', 'ICP', 'Surgery History', 'Hypothermia Therapy',
                          'Albumin', 'Blood Glucose', 'Mechanical Ventilation', 'Feeding Route', 
                          'EN Start Time', 'IAP', 'AGI Grade', 'Sedation Depth', 'Vasoactive Drugs', 'Diabetes History']
            for label, val in zip(param_labels, input_data[0]):
                param_data.append([label, f'{val:.1f}' if isinstance(val, float) else str(int(val))])
            
            param_table = Table(param_data, colWidths=[3*inch, 2*inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(param_table)
            story.append(Spacer(1, 15))
            
            # Feature Contributions
            story.append(Paragraph("Feature Contributions", heading_style))
            contrib_data = [['Feature', 'Value', 'Contribution', 'Direction']]
            for _, row in contrib_df.head(10).iterrows():
                contrib_data.append([row['Feature'], f'{row["Value"]:.2f}', f'{row["Contribution"]:.4f}', row['Direction']])
            
            contrib_table = Table(contrib_data, colWidths=[2*inch, 1*inch, 1.2*inch, 1.3*inch])
            contrib_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            story.append(contrib_table)
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("<i>This report was generated by the ENI Risk Prediction System V1.0. The prediction is based on machine learning models and should be used as a clinical decision support tool, not as a substitute for professional medical judgment.</i>", ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey)))
            
            doc.build(story)
            pdf = buffer.getvalue()
            buffer.close()
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf,
                file_name=f"ENI_Risk_Report_{age}yo.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:12px;">
    <p>ENI Risk Prediction System V1.0 | Developed for Clinical Decision Support</p>
    <p>Based on Lasso-Logistic Regression Model | Internal Bootstrap Validation AUC: 0.925</p>
</div>
""", unsafe_allow_html=True)
