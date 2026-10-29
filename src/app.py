
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Keep the existing backend and prediction pipeline unchanged.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import DISEASE_CONFIGS
from prediction import predict_risk


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HealthScope – Personal Health Risk Assessment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# USER-FACING DESIGN
# ============================================================
CUSTOM_CSS = """
<style>
:root {
    --navy: #123B78;
    --navy-dark: #0B2D5C;
    --teal: #18A999;
    --teal-light: #EAF9F6;
    --blue-light: #EEF5FC;
    --bg: #F6F9FC;
    --white: #FFFFFF;
    --text: #10233F;
    --muted: #607089;
    --border: #DDE7F1;
    --green: #22A06B;
    --green-bg: #ECF9F2;
    --orange: #F2A51A;
    --orange-bg: #FFF7E8;
    --red: #E45757;
    --red-bg: #FFF0F0;
}

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stSidebar"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: rgba(246,249,252,0.95);
}

.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* Top navigation */
.top-nav {
    background: rgba(255,255,255,0.97);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 13px 20px;
    margin-bottom: 24px;
    box-shadow: 0 4px 18px rgba(18,59,120,0.05);
}

.brand {
    color: var(--navy) !important;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.nav-link {
    color: #38516F !important;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    padding-top: 7px;
}

.nav-pill {
    background: var(--navy);
    color: white !important;
    border-radius: 9px;
    padding: 9px 17px;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
}

/* Hero */
.hero {
    background: linear-gradient(105deg, #FFFFFF 0%, #F7FBFF 62%, #EAF4FB 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 50px;
    margin-bottom: 28px;
    box-shadow: 0 8px 30px rgba(18,59,120,0.06);
}

.hero-kicker {
    color: var(--teal) !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero h1 {
    color: var(--navy-dark) !important;
    font-size: 46px;
    line-height: 1.04;
    letter-spacing: -1.8px;
    margin: 0 0 15px 0;
    font-weight: 800;
}

.hero p {
    color: #4E627D !important;
    font-size: 16px;
    line-height: 1.65;
    max-width: 650px;
    margin: 0;
}

/* Section headings */
.section-title {
    color: var(--navy-dark) !important;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 26px 0 7px 0;
}

.section-subtitle {
    color: var(--muted) !important;
    font-size: 14px;
    margin-bottom: 16px;
}

/* Condition cards */
.condition-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(18,59,120,0.045);
}

.condition-name {
    color: var(--navy-dark) !important;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 7px;
}

.condition-text {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.45;
}

/* Forms */
[data-testid="stForm"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 26px 28px 22px 28px !important;
    box-shadow: 0 7px 25px rgba(18,59,120,0.05);
}

[data-testid="stForm"] label {
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 650 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: white !important;
    border-color: #CAD8E6 !important;
    border-radius: 9px !important;
}

input {
    color: var(--text) !important;
}

div[data-baseweb="select"] * {
    color: var(--text) !important;
}

/* Buttons */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--navy) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 9px !important;
    min-height: 43px !important;
    font-weight: 750 !important;
    box-shadow: none !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--navy-dark) !important;
    color: white !important;
}

/* Result cards */
.result-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px;
    min-height: 270px;
    box-shadow: 0 7px 25px rgba(18,59,120,0.05);
}

.result-card.low {
    border-left: 5px solid var(--green);
    background: linear-gradient(135deg, #FFFFFF, #F4FCF8);
}

.result-card.moderate {
    border-left: 5px solid var(--orange);
    background: linear-gradient(135deg, #FFFFFF, #FFFBF3);
}

.result-card.high {
    border-left: 5px solid var(--red);
    background: linear-gradient(135deg, #FFFFFF, #FFF7F7);
}

.small-label {
    color: var(--muted) !important;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}

.risk-label {
    color: var(--navy-dark) !important;
    font-size: 32px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-bottom: 18px;
}

.probability {
    color: var(--navy-dark) !important;
    font-size: 30px;
    font-weight: 800;
}

.info-banner {
    background: var(--blue-light);
    border: 1px solid #D4E4F5;
    color: #31506F !important;
    border-radius: 12px;
    padding: 15px 18px;
    font-size: 13px;
    line-height: 1.5;
    margin: 12px 0 24px 0;
}

/* Parameter cards */
.parameter-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px 18px;
    min-height: 112px;
    box-shadow: 0 4px 15px rgba(18,59,120,0.04);
}

.parameter-name {
    color: #63748B !important;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 7px;
}

.parameter-value {
    color: var(--navy-dark) !important;
    font-size: 20px;
    font-weight: 800;
}

.parameter-status {
    font-size: 11px;
    font-weight: 750;
    margin-top: 7px;
}

/* Insight cards */
.insight-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    min-height: 105px;
    box-shadow: 0 4px 15px rgba(18,59,120,0.035);
}

.insight-title {
    color: var(--navy) !important;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 7px;
}

.insight-text {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.45;
}

/* Next steps */
.next-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 4px 15px rgba(18,59,120,0.035);
}

.next-title {
    color: var(--navy-dark) !important;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 7px;
}

.next-text {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.45;
}

/* Chart containers */
[data-testid="stImage"],
[data-testid="stPyplot"] {
    background: white !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 7px;
    box-shadow: 0 4px 15px rgba(18,59,120,0.035);
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Footer */
.footer {
    margin-top: 45px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    text-align: center;
    color: #718096 !important;
    font-size: 11px;
    line-height: 1.5;
}

@media (max-width: 800px) {
    .hero { padding: 32px 25px; }
    .hero h1 { font-size: 36px; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = "Diabetes"

if "assessment_data" not in st.session_state:
    st.session_state.assessment_data = None


# ============================================================
# CHARTS
# ============================================================
def plot_risk_gauge(probability):
    """Clean semicircle gauge for the user's estimated probability."""
    probability = float(np.clip(probability, 0, 100))

    fig, ax = plt.subplots(figsize=(5.1, 2.85))
    theta = np.linspace(np.pi, 0, 240)

    # Green / amber / red zones
    ax.plot(theta[0:84], np.ones(84), lw=24, color="#36B37E", solid_capstyle="round")
    ax.plot(theta[84:156], np.ones(72), lw=24, color="#F5B642", solid_capstyle="butt")
    ax.plot(theta[156:], np.ones(84), lw=24, color="#E95B5B", solid_capstyle="round")

    value_theta = np.pi - (probability / 100.0) * np.pi
    ax.annotate(
        "",
        xy=(value_theta, 1.0),
        xytext=(np.pi / 2, 0.15),
        arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#10233F"),
    )

    ax.text(
        0.5, 0.28, f"{probability:.1f}%",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=23, fontweight="800", color="#10233F"
    )
    ax.text(
        0.5, 0.08, "Estimated Risk",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, color="#607089"
    )

    ax.set_xlim(-0.15, np.pi + 0.15)
    ax.set_ylim(0, 1.18)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.3)
    return fig


def plot_reference_comparison(value, label, unit, low, high):
    """Shows the user's value against a meaningful reference interval."""
    value = float(value)
    low = float(low)
    high = float(high)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axvspan(low, high, color="#DDF4EA", alpha=0.95)
    ax.axvline(low, color="#36B37E", lw=1.5)
    ax.axvline(high, color="#36B37E", lw=1.5)
    ax.scatter([value], [0], s=130, color="#123B78", zorder=5)

    ax.set_xlim(max(0, min(low * 0.75, value * 0.75)), max(high * 1.25, value * 1.25))
    ax.set_yticks([])
    ax.set_xlabel(f"{label} ({unit})", color="#475569")
    ax.set_title(f"{label} vs Reference Range", color="#123B78", fontweight="800")
    ax.grid(axis="x", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#DDE7F1")

    ax.text(value, 0.14, f"{value:g}", ha="center", color="#123B78", fontweight="800")
    ax.text(low, -0.12, f"{low:g}", transform=ax.get_xaxis_transform(),
            ha="center", color="#607089", fontsize=9)
    ax.text(high, -0.12, f"{high:g}", transform=ax.get_xaxis_transform(),
            ha="center", color="#607089", fontsize=9)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def plot_bmi_donut_chart(user_bmi):
    """BMI category visualization without the previous pie-unpacking error."""
    bmi = float(user_bmi)

    categories = ["Underweight", "Normal", "Overweight", "Obese"]
    sizes = [18.5, 6.4, 5.0, 10.0]
    colors = ["#46B9E8", "#36C275", "#F5B642", "#E95B5B"]

    fig, ax = plt.subplots(figsize=(5.1, 3.35))
    ax.pie(
        sizes,
        labels=categories,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.38, edgecolor="white"),
        textprops=dict(fontsize=9),
    )
    ax.text(
        0, 0, f"{bmi:.1f}\nkg/m²",
        ha="center", va="center",
        fontsize=16, fontweight="800", color="#123B78"
    )
    ax.set_title("BMI Category", color="#123B78", fontweight="800", pad=8)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def plot_parameter_profile(items):
    """Readable comparison chart based only on entered user values."""
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    max_values = [x[2] for x in items]

    normalized = [
        min(100, max(0, (float(v) / float(m)) * 100))
        for v, m in zip(values, max_values)
    ]

    fig, ax = plt.subplots(figsize=(5.4, 3.35))
    y = np.arange(len(labels))

    ax.barh(y, [100] * len(labels), color="#EAF1F7", height=0.48)
    ax.barh(y, normalized, color="#18A999", height=0.48)

    for idx, pct in enumerate(normalized):
        ax.text(
            min(pct + 2, 94), idx, f"{pct:.0f}%",
            va="center", fontsize=9, fontweight="700", color="#123B78"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative position within input scale")
    ax.set_title("Health Parameter Comparison", color="#123B78", fontweight="800")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#DDE7F1")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ============================================================
# HELPERS
# ============================================================
def risk_class(probability):
    if probability < 35:
        return "LOWER RISK", "low"
    if probability < 65:
        return "MODERATE RISK", "moderate"
    return "HIGHER RISK", "high"


def value_status(label, value):
    """Educational reference labels for the result cards."""
    v = float(value)

    if label == "Glucose":
        if v >= 126:
            return "Elevated", "#E45757"
        if v >= 100:
            return "Above typical range", "#F2A51A"
        return "Normal", "#22A06B"

    if label == "Blood Pressure":
        if v >= 130:
            return "Elevated", "#F2A51A"
        return "Within reference range", "#22A06B"

    if label == "BMI":
        if v < 18.5:
            return "Underweight", "#3CAFE0"
        if v < 25:
            return "Normal", "#22A06B"
        if v < 30:
            return "Overweight", "#F2A51A"
        return "Obese range", "#E45757"

    if label == "Cholesterol":
        if v >= 240:
            return "High", "#E45757"
        if v >= 200:
            return "Borderline high", "#F2A51A"
        return "Within reference range", "#22A06B"

    if label == "Systolic BP":
        if v >= 130:
            return "Elevated", "#F2A51A"
        return "Within reference range", "#22A06B"

    if label == "Diastolic BP":
        if v >= 85:
            return "Elevated", "#F2A51A"
        return "Within reference range", "#22A06B"

    if label == "Age":
        return "Adult", "#123B78"

    if label == "Heart Rate":
        if 60 <= v <= 100:
            return "Typical resting range", "#22A06B"
        return "Outside typical resting range", "#F2A51A"

    return "Entered value", "#123B78"


def parameter_card(label, value, status=None, color="#123B78"):
    status_html = ""
    if status:
        status_html = f'<div class="parameter-status" style="color:{color};">{status}</div>'

    st.markdown(
        f"""
        <div class="parameter-card">
            <div class="parameter-name">{label}</div>
            <div class="parameter-value">{value}</div>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_parameter_cards(disease_key, inputs):
    cards = []

    if disease_key == "diabetes":
        glucose = inputs["Glucose"]
        bmi = inputs["BMI"]
        bp = inputs["BloodPressure"]
        age = inputs["Age"]

        s, c = value_status("Glucose", glucose)
        cards.append(("Glucose", f"{glucose:g} mg/dL", s, c))
        s, c = value_status("Blood Pressure", bp)
        cards.append(("Blood Pressure", f"{bp:g} mmHg", s, c))
        s, c = value_status("BMI", bmi)
        cards.append(("BMI", f"{bmi:.1f} kg/m²", s, c))
        s, c = value_status("Age", age)
        cards.append(("Age", f"{age} years", s, c))

    elif disease_key == "hypertension":
        sbp = inputs["sysBP"]
        dbp = inputs["diaBP"]
        bmi = inputs["BMI"]
        age = inputs["age"]

        s, c = value_status("Systolic BP", sbp)
        cards.append(("Systolic BP", f"{sbp:g} mmHg", s, c))
        s, c = value_status("Diastolic BP", dbp)
        cards.append(("Diastolic BP", f"{dbp:g} mmHg", s, c))
        s, c = value_status("BMI", bmi)
        cards.append(("BMI", f"{bmi:.1f} kg/m²", s, c))
        s, c = value_status("Age", age)
        cards.append(("Age", f"{age} years", s, c))

    else:
        age = inputs["age"]
        bp = inputs["trestbps"]
        chol = inputs["chol"]
        hr = inputs["thalach"]

        s, c = value_status("Blood Pressure", bp)
        cards.append(("Blood Pressure", f"{bp:g} mmHg", s, c))
        s, c = value_status("Cholesterol", chol)
        cards.append(("Cholesterol", f"{chol:g} mg/dL", s, c))
        s, c = value_status("Heart Rate", hr)
        cards.append(("Max Heart Rate", f"{hr:g} bpm", s, c))
        s, c = value_status("Age", age)
        cards.append(("Age", f"{age} years", s, c))

    return cards


def health_insights(disease_key, inputs, risk_label):
    insights = []

    if disease_key == "diabetes":
        if inputs["Glucose"] >= 126:
            insights.append("Your blood glucose is above the reference range used in this assessment.")
        elif inputs["Glucose"] >= 100:
            insights.append("Your blood glucose is above the typical baseline range used here.")
        else:
            insights.append("Your blood glucose is within the typical baseline range used here.")

        if inputs["BMI"] >= 25:
            insights.append("Your BMI falls within the overweight or higher range.")
        else:
            insights.append("Your BMI is within the normal reference range.")

        if inputs["BloodPressure"] >= 130:
            insights.append("Your blood pressure reading is above the reference range used for this assessment.")
        else:
            insights.append("Your blood pressure is within the reference range used for this assessment.")

    elif disease_key == "hypertension":
        if inputs["sysBP"] >= 130:
            insights.append("Your systolic blood pressure is elevated compared with the reference range used here.")
        else:
            insights.append("Your systolic blood pressure is within the reference range used here.")

        if inputs["diaBP"] >= 85:
            insights.append("Your diastolic blood pressure is above the reference range used here.")
        else:
            insights.append("Your diastolic blood pressure is within the reference range used here.")

        if inputs["BMI"] >= 25:
            insights.append("Your BMI falls within the overweight or higher range.")
        else:
            insights.append("Your BMI is within the normal reference range.")

    else:
        if inputs["trestbps"] >= 130:
            insights.append("Your resting blood pressure is elevated compared with the reference range used here.")
        else:
            insights.append("Your resting blood pressure is within the reference range used here.")

        if inputs["chol"] >= 240:
            insights.append("Your cholesterol value is in the high range used for this educational comparison.")
        elif inputs["chol"] >= 200:
            insights.append("Your cholesterol value is above the preferred reference level used here.")
        else:
            insights.append("Your cholesterol value is within the preferred reference range used here.")

        if inputs["exang"] == 1:
            insights.append("Exercise-induced chest discomfort was included in your assessment inputs.")

    if risk_label == "HIGHER RISK":
        insights.append("The assessment indicates a higher estimated risk; consider discussing these results with a qualified healthcare professional.")
    elif risk_label == "MODERATE RISK":
        insights.append("The assessment indicates a moderate estimated risk; reviewing modifiable health factors may be useful.")
    else:
        insights.append("The assessment indicates a lower estimated risk based on the information entered.")

    return insights[:4]


def next_steps(disease_key, inputs):
    if disease_key == "diabetes":
        return [
            ("Balanced nutrition", "Choose balanced meals with vegetables, whole grains, fiber-rich foods and moderate added sugars."),
            ("Regular activity", "Aim for consistent physical activity that fits your current fitness level."),
            ("Monitor health measures", "Keep track of relevant health measurements and discuss unusual changes with a healthcare professional."),
        ]

    if disease_key == "hypertension":
        return [
            ("Heart-healthy eating", "Choose more fresh foods and be mindful of excess sodium in everyday meals."),
            ("Regular activity", "Maintain a consistent, moderate physical activity routine where appropriate."),
            ("Monitor blood pressure", "Track blood pressure readings and discuss persistently elevated values with a healthcare professional."),
        ]

    return [
        ("Heart-healthy nutrition", "Choose balanced meals with vegetables, fruits, whole grains and lean protein sources."),
        ("Regular activity", "Maintain a consistent activity routine that is appropriate for your health and fitness level."),
        ("Monitor health measures", "Keep track of relevant measurements such as blood pressure and cholesterol and review them with a professional."),
    ]


# ============================================================
# HEADER
# ============================================================
def render_nav():
    st.markdown(
        """
        <div class="top-nav">
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns([2.5, 1, 1.2, 0.8, 1.1])
    with c1:
        st.markdown('<div class="brand">HealthScope</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nav-link">Home</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="nav-link">Health Assessment</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="nav-link">About</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="nav-pill">Get Started</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================
def render_assessment_form():
    st.markdown(
        """
        <div class="section-title">Health Risk Assessment</div>
        <div class="section-subtitle">
            Enter your information below to receive a personalized educational risk assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_condition = st.selectbox(
        "What would you like to assess?",
        ["Diabetes", "Heart Disease", "Hypertension"],
        index=["Diabetes", "Heart Disease", "Hypertension"].index(
            st.session_state.selected_condition
        ),
    )

    st.session_state.selected_condition = selected_condition
    disease_key = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart_disease",
        "Hypertension": "hypertension",
    }[selected_condition]

    with st.form("healthscope_assessment"):
        st.markdown(
            f"""
            <div style="font-size:17px;font-weight:800;color:#123B78;margin-bottom:16px;">
                Enter Your Health Information
            </div>
            """,
            unsafe_allow_html=True,
        )

        input_dict = {}

        if disease_key == "diabetes":
            col1, col2 = st.columns(2)
            with col1:
                input_dict["Age"] = st.number_input("Age (years)", 1, 120, 35, 1)
                input_dict["Glucose"] = st.number_input("Blood Glucose (mg/dL)", 0, 300, 120, 1)
                input_dict["BloodPressure"] = st.number_input("Blood Pressure (mmHg)", 0, 200, 70, 1)
                input_dict["BMI"] = st.number_input("Body Mass Index (kg/m²)", 0.0, 80.0, 28.4, 0.1)
                input_dict["Pregnancies"] = st.number_input("Pregnancies (count)", 0, 20, 1, 1)

            with col2:
                st.markdown(
                    '<div style="color:#607089;font-size:12px;margin-bottom:8px;">Optional clinical information</div>',
                    unsafe_allow_html=True,
                )
                know_tests = st.checkbox("I have recent diabetes test results", value=False)
                if know_tests:
                    input_dict["SkinThickness"] = st.number_input(
                        "Skin Thickness (mm)", 0, 100, 20, 1
                    )
                    input_dict["Insulin"] = st.number_input(
                        "Insulin (µU/mL)", 0, 1000, 80, 1
                    )
                    input_dict["DiabetesPedigreeFunction"] = st.number_input(
                        "Diabetes Family History Score", 0.0, 3.5, 0.45, 0.01
                    )
                else:
                    # Baseline values keep the existing prediction pipeline compatible
                    # when the user does not have these less-familiar measurements.
                    input_dict["SkinThickness"] = 20
                    input_dict["Insulin"] = 80
                    input_dict["DiabetesPedigreeFunction"] = 0.45

        elif disease_key == "heart_disease":
            col1, col2 = st.columns(2)
            with col1:
                input_dict["age"] = st.number_input("Age (years)", 1, 120, 55, 1)
                input_dict["sex"] = st.selectbox(
                    "Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female"
                )
                input_dict["cp"] = st.selectbox(
                    "Chest Pain Type",
                    [0, 1, 2, 3],
                    format_func=lambda x: [
                        "Typical Angina", "Atypical Angina",
                        "Non-anginal Pain", "Asymptomatic"
                    ][x],
                )
                input_dict["trestbps"] = st.number_input(
                    "Resting Blood Pressure (mmHg)", 50, 250, 130, 1
                )
                input_dict["chol"] = st.number_input(
                    "Serum Cholesterol (mg/dL)", 80, 600, 245, 1
                )
                st.markdown(
                    '<div style="font-size:14px;font-weight:750;color:#123B78;margin:8px 0 10px;">Additional Health Information</div>',
                    unsafe_allow_html=True,
                )
                input_dict["fbs"] = st.selectbox(
                    "Fasting blood sugar above 120 mg/dL?",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )
                input_dict["exang"] = st.selectbox(
                    "Chest discomfort during exercise?",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )
                input_dict["thalach"] = st.number_input(
                    "Maximum heart rate (bpm) — if known",
                    50, 230, 150, 1,
                )

            with col2:
                st.markdown(
                    '<div style="font-size:14px;font-weight:750;color:#123B78;margin:8px 0 10px;">Medical Test Results</div>',
                    unsafe_allow_html=True,
                )
                st.caption("These values are optional. If you do not have recent test results, the assessment will use standard baseline values.")
                know_tests = st.checkbox("I have recent heart test results", value=False)

                if know_tests:
                    input_dict["restecg"] = st.selectbox(
                        "Resting ECG result",
                        [0, 1, 2],
                        format_func=lambda x: [
                            "Normal", "ST-T Wave Abnormality", "LV Hypertrophy"
                        ][x],
                    )
                    input_dict["oldpeak"] = st.number_input(
                        "ST depression during exercise",
                        0.0, 10.0, 1.0, 0.1,
                    )
                    input_dict["slope"] = st.selectbox(
                        "ST segment slope",
                        [0, 1, 2],
                        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                    )
                    input_dict["ca"] = st.number_input(
                        "Major vessels shown by imaging (0–4)",
                        0, 4, 0, 1,
                    )
                    input_dict["thal"] = st.selectbox(
                        "Thalassemia test result",
                        [0, 1, 2, 3],
                        format_func=lambda x: [
                            "Normal", "Fixed Defect", "Reversible Defect", "Severe"
                        ][x],
                    )
                else:
                    # Preserve the model's required feature vector without
                    # forcing users to know specialized clinical terminology.
                    input_dict["restecg"] = 0
                    input_dict["oldpeak"] = 0.0
                    input_dict["slope"] = 1
                    input_dict["ca"] = 0
                    input_dict["thal"] = 2

        else:
            col1, col2 = st.columns(2)
            with col1:
                input_dict["age"] = st.number_input("Age (years)", 1, 120, 52, 1)
                input_dict["male"] = st.selectbox(
                    "Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female"
                )
                input_dict["sysBP"] = st.number_input(
                    "Systolic Blood Pressure (mmHg)", 60, 300, 138, 1
                )
                input_dict["diaBP"] = st.number_input(
                    "Diastolic Blood Pressure (mmHg)", 30, 200, 88, 1
                )
                input_dict["BMI"] = st.number_input(
                    "Body Mass Index (kg/m²)", 10.0, 80.0, 28.4, 0.1
                )
                input_dict["heartRate"] = st.number_input(
                    "Heart Rate (bpm)", 30, 200, 75, 1
                )

            with col2:
                input_dict["currentSmoker"] = st.selectbox(
                    "Do you currently smoke?", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                )
                input_dict["diabetes"] = st.selectbox(
                    "Have you been diagnosed with diabetes?", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                )

                st.markdown(
                    '<div style="color:#607089;font-size:12px;margin:12px 0 8px;">Optional medical information</div>',
                    unsafe_allow_html=True,
                )
                know_medical = st.checkbox("I know my other health measurements", value=False)
                if know_medical:
                    input_dict["totChol"] = st.number_input(
                        "Total Cholesterol (mg/dL)", 80, 600, 235, 1
                    )
                    input_dict["glucose"] = st.number_input(
                        "Glucose Level (mg/dL)", 30, 500, 85, 1
                    )
                    input_dict["cigsPerDay"] = st.number_input(
                        "Cigarettes Per Day", 0, 100, 0, 1
                    )
                    input_dict["BPMeds"] = st.selectbox(
                        "Taking Blood Pressure Medication?", [0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )
                    input_dict["prevalentStroke"] = st.selectbox(
                        "Previous Stroke?", [0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )
                else:
                    # Baseline values preserve the existing model input shape.
                    input_dict["totChol"] = 235
                    input_dict["glucose"] = 85
                    input_dict["cigsPerDay"] = 0
                    input_dict["BPMeds"] = 0
                    input_dict["prevalentStroke"] = 0

        submitted = st.form_submit_button("Assess My Risk", width="stretch")

    if submitted:
        try:
            result = predict_risk(disease_key, input_dict)

            st.session_state.assessment_data = {
                "result": result,
                "inputs": input_dict,
                "condition_name": selected_condition,
                "disease_key": disease_key,
            }

        except Exception:
            st.session_state.assessment_data = None
            st.error(
                "We couldn't complete the assessment with the information provided. "
                "Please review your entries and try again."
            )


# ============================================================
# RESULTS
# ============================================================
def render_results():
    data = st.session_state.assessment_data
    result = data["result"]
    inputs = data["inputs"]
    condition = data["condition_name"]
    disease_key = data["disease_key"]

    probability = result.get("probability")
    if probability is None:
        probability = 75.0 if result.get("raw_pred") == 1 else 25.0

    risk_label, risk_class_name = risk_class(probability)

    st.markdown(
        f"""
        <div style="margin-top:40px;">
            <div class="section-title" style="font-size:31px;">
                Your {condition} Risk Assessment
            </div>
            <div class="section-subtitle">
                Here is your personalized educational health risk assessment based on the information provided.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns([1.02, 0.98])

    with r1:
        st.markdown(
            f"""
            <div class="result-card {risk_class_name}">
                <div class="small-label">Risk Level</div>
                <div class="risk-label">{risk_label.title()}</div>
                <div class="small-label">Estimated Risk Probability</div>
                <div class="probability">{probability:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.pyplot(plot_risk_gauge(probability), width="stretch")

    st.markdown(
        """
        <div class="info-banner">
            This result is an educational risk estimate based on the information you entered.
            It is not a medical diagnosis and should not replace professional medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Contributing parameters
    # -----------------------------
    st.markdown(
        '<div class="section-title">What May Be Contributing to Your Result</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Your entered values compared with common reference ranges.</div>',
        unsafe_allow_html=True,
    )

    cards = build_parameter_cards(disease_key, inputs)
    cols = st.columns(4)
    for col, card in zip(cols, cards):
        with col:
            parameter_card(*card)

    # -----------------------------
    # Visual insights
    # -----------------------------
    st.markdown(
        '<div class="section-title">Visual Health Insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Visual comparisons based on the health information you entered.</div>',
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(3)

    if disease_key == "diabetes":
        with chart_cols[0]:
            st.pyplot(
                plot_reference_comparison(
                    inputs["Glucose"], "Glucose", "mg/dL", 70, 99
                ),
                width="stretch",
            )
        with chart_cols[1]:
            st.pyplot(
                plot_bmi_donut_chart(inputs["BMI"]),
                width="stretch",
            )
        with chart_cols[2]:
            items = [
                ("Glucose", inputs["Glucose"], 200),
                ("BMI", inputs["BMI"], 40),
                ("Blood Pressure", inputs["BloodPressure"], 180),
                ("Age", inputs["Age"], 100),
            ]
            st.pyplot(plot_parameter_profile(items), width="stretch")

    elif disease_key == "hypertension":
        with chart_cols[0]:
            st.pyplot(
                plot_reference_comparison(
                    inputs["sysBP"], "Systolic BP", "mmHg", 90, 120
                ),
                width="stretch",
            )
        with chart_cols[1]:
            st.pyplot(
                plot_bmi_donut_chart(inputs["BMI"]),
                width="stretch",
            )
        with chart_cols[2]:
            items = [
                ("Systolic BP", inputs["sysBP"], 180),
                ("Diastolic BP", inputs["diaBP"], 120),
                ("Cholesterol", inputs["totChol"], 300),
                ("Age", inputs["age"], 100),
            ]
            st.pyplot(plot_parameter_profile(items), width="stretch")

    else:
        with chart_cols[0]:
            st.pyplot(
                plot_reference_comparison(
                    inputs["trestbps"], "Resting BP", "mmHg", 90, 120
                ),
                width="stretch",
            )
        with chart_cols[1]:
            st.pyplot(
                plot_reference_comparison(
                    inputs["chol"], "Cholesterol", "mg/dL", 125, 200
                ),
                width="stretch",
            )
        with chart_cols[2]:
            items = [
                ("Blood Pressure", inputs["trestbps"], 200),
                ("Cholesterol", inputs["chol"], 400),
                ("Max Heart Rate", inputs["thalach"], 220),
                ("Age", inputs["age"], 100),
            ]
            st.pyplot(plot_parameter_profile(items), width="stretch")

    # -----------------------------
    # Key health insights
    # -----------------------------
    st.markdown(
        '<div class="section-title">Key Health Insights</div>',
        unsafe_allow_html=True,
    )

    insights = health_insights(disease_key, inputs, risk_label)
    insight_cols = st.columns(2)

    for i, insight in enumerate(insights):
        with insight_cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Health Insight</div>
                    <div class="insight-text">{insight}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------
    # Next steps
    # -----------------------------
    st.markdown(
        '<div class="section-title">Suggested Next Steps</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">General educational measures that may support healthier habits.</div>',
        unsafe_allow_html=True,
    )

    steps = next_steps(disease_key, inputs)
    step_cols = st.columns(3)

    for col, (title, text) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="next-card">
                    <div class="next-title">{title}</div>
                    <div class="next-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer">
            This assessment is for educational purposes only and is not a medical diagnosis.
            Consult a qualified healthcare professional for medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    render_nav()

    # Landing / assessment introduction
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Personal Health Assessment</div>
            <h1>Your Health<br>In Focus</h1>
            <p>
                Answer a few simple health questions and receive a clear, educational risk assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick condition overview
    condition_cols = st.columns(3)
    condition_data = [
        (
            "Diabetes Risk",
            "Understand your estimated diabetes risk using relevant health indicators.",
        ),
        (
            "Heart Disease Risk",
            "Explore your estimated heart disease risk using clinical information.",
        ),
        (
            "Hypertension Risk",
            "Review your estimated hypertension risk using key health parameters.",
        ),
    ]

    for col, (title, description) in zip(condition_cols, condition_data):
        with col:
            st.markdown(
                f"""
                <div class="condition-card">
                    <div class="condition-name">{title}</div>
                    <div class="condition-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_assessment_form()

    if st.session_state.assessment_data is not None:
        render_results()


if __name__ == "__main__":
    main()
