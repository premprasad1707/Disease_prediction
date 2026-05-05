import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import list_trained_diseases, load_model, predict_single
from src.chatbot import analyze_symptoms, format_bot_response

# ─── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | Premium Multi-Disease Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM STYLING ───────────────────────────────────────────────
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #020617);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card-like containers (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #818cf8;
        font-weight: 700;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #6366f1;
    }
    
    /* Remove default streamlit header */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

local_css()

# ─── CONSTANTS ────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This AI tool is for educational and screening assistance only. "
    "It is NOT a replacement for professional medical diagnosis or advice. "
    "In case of emergency, contact your local healthcare provider immediately."
)

# ─── DATA & MODELS ────────────────────────────────────────────────
diseases = list_trained_diseases()

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 class='gradient-text' style='text-align: center; font-size: 2.5rem;'>🛡️ MediScan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Disease Prediction AI</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("NAVIGATION", [
        "🏠 Dashboard",
        "🔬 Prediction Center",
        "💬 Symptom AI Chat",
        "📊 Model Insights"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("System Status")
    st.success(f"Models Active: {len(diseases)}")
    st.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    with st.expander("⚕️ Legal Disclaimer"):
        st.caption(DISCLAIMER)

# ─── PAGE 1: DASHBOARD ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("<h1 class='gradient-text'>Global Health Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Welcome to MediScan AI. Monitor, predict, and analyze health risks with state-of-the-art ML models.")
    
    if not diseases:
        st.error("No models found. Please train the system using `src/train_all.py`.")
        st.stop()

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">Active Models</p><h2 style="margin:0;">{len(diseases)}</h2></div>""", unsafe_allow_html=True)
    with c2:
        avg_acc = sum(d['accuracy'] for d in diseases) / len(diseases)
        st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">Avg Accuracy</p><h2 style="margin:0;">{avg_acc*100:.1f}%</h2></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">Dataset Coverage</p><h2 style="margin:0;">Global</h2></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">AI Engine</p><h2 style="margin:0;">Hybrid</h2></div>""", unsafe_allow_html=True)

    st.markdown("### Available AI Diagnostic Modules")
    
    # Disease Cards
    cols = st.columns(3)
    for i, d in enumerate(diseases):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size: 2.5rem;">{d['icon']}</span>
                    <div>
                        <h3 style="margin:0;">{d['display_name']}</h3>
                        <p style="color:#818cf8; margin:0; font-weight:600;">Accuracy: {d['accuracy']*100:.1f}%</p>
                    </div>
                </div>
                <p style="color:#94a3b8; margin-top:12px; font-size:0.9rem;">
                    Powered by <b>{d['best_model']}</b>. Optimized for precision and low false-negative rates.
                </p>
                <div style="text-align:right;">
                    <p style="font-size:0.8rem; color:#6366f1;">ROC-AUC: {d['roc_auc']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Access")
    q1, q2 = st.columns(2)
    with q1:
        if st.button("🚀 Start New Prediction"):
            st.info("Select 'Prediction Center' from the sidebar to begin.")
    with q2:
        if st.button("💬 Consult Symptom AI"):
            st.info("Select 'Symptom AI Chat' from the sidebar.")


# ─── PAGE 2: PREDICTION CENTER ──────────────────────────────────────
elif page == "🔬 Prediction Center":
    st.markdown("<h1 class='gradient-text'>Precision Diagnostic Center</h1>", unsafe_allow_html=True)
    st.markdown("Input clinical data to generate a detailed risk assessment report.")

    if not diseases:
        st.warning("No trained models available.")
        st.stop()

    disease_names = [f"{d['icon']} {d['display_name']}" for d in diseases]
    
    col_sel, col_empty = st.columns([1, 1])
    with col_sel:
        selected = st.selectbox("Select Target Condition", disease_names)
    
    selected_disease = diseases[[f"{d['icon']} {d['display_name']}" for d in diseases].index(selected)]
    safe_name = selected_disease["safe_name"]

    try:
        model, scaler, meta = load_model(safe_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    features = meta["features"]
    feature_stats = meta.get("feature_stats", {})
    is_multiclass = meta.get("is_multiclass", False)

    st.markdown(f"### {selected_disease['icon']} Patient Data Input: {selected_disease['display_name']}")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        input_dict = {}
        
        # Split features into 3 columns
        num_cols = 3
        feat_cols = st.columns(num_cols)
        
        for i, feat in enumerate(features):
            col = feat_cols[i % num_cols]
            stats = feature_stats.get(feat, {})
            min_v = float(stats.get("min", 0.0))
            max_v = float(stats.get("max", 100.0))
            mean_v = float(stats.get("mean", 0.0))
            unique = int(stats.get("unique", 999))
            
            label = feat.replace("_", " ").title()
            
            with col:
                if unique <= 2:
                    val = st.selectbox(label, [0, 1], 
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       key=f"feat_{feat}_{safe_name}")
                elif max_v - min_v <= 1.0 and max_v <= 1.01:
                    # Likely a probability or normalized value
                    val = st.slider(label, min_v, max_v, mean_v, step=0.01, key=f"feat_{feat}_{safe_name}")
                else:
                    # Standard numerical
                    val = st.number_input(label, min_v, max_v, mean_v, key=f"feat_{feat}_{safe_name}")
                
                input_dict[feat] = val
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(f"🔍 Generate {selected_disease['display_name']} Assessment Report"):
        with st.spinner("AI Engine analyzing clinical data..."):
            result = predict_single(safe_name, input_dict)

        st.markdown("---")
        st.markdown("<h2 class='gradient-text'>Diagnostic Results</h2>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            prob = result["probability"]
            risk = result["risk_level"]
            severity = result["severity"]
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score (%)", 'font': {'size': 24, 'color': '#f8fafc'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#6366f1"},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.3)'},
                        {'range': [35, 65], 'color': 'rgba(234, 179, 8, 0.3)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ],
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc", 'family': "Inter"})
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown(f'<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            if is_multiclass:
                st.subheader(f"Top Likely Condition: {result['prediction_label']}")
                st.markdown(f"**Confidence:** {prob}%")
                
                st.markdown("#### Differential Diagnosis")
                for tc in result.get("top_classes", []):
                    st.markdown(f"- **{tc['label']}**: {tc['prob']}%")
                    st.progress(int(tc['prob']))
            else:
                pred_text = "DISEASE DETECTED" if result["prediction"] == 1 else "NO DISEASE DETECTED"
                pred_color = "#ef4444" if result["prediction"] == 1 else "#22c55e"
                st.markdown(f"<h1 style='color:{pred_color}; text-align:center;'>{pred_text}</h1>", unsafe_allow_html=True)
                st.markdown(f"**Risk Level:** {risk}")
            
            st.markdown("---")
            st.markdown(f"💡 **AI Message:** {result['message']}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Health Recommendations")
        rec_cols = st.columns(len(result["recommendations"]) if result["recommendations"] else 1)
        for i, rec in enumerate(result["recommendations"]):
            with rec_cols[i % len(rec_cols)]:
                st.markdown(f"""<div class="glass-card" style="padding:15px; border-left: 4px solid #818cf8;">{rec}</div>""", unsafe_allow_html=True)
        
        st.caption(DISCLAIMER)


# ─── PAGE 3: CHATBOT ──────────────────────────────────────────────
elif page == "💬 Symptom AI Chat":
    st.markdown("<h1 class='gradient-text'>Symptom AI Consultant</h1>", unsafe_allow_html=True)
    st.markdown("Describe how you're feeling in plain English. Our AI will analyze your symptoms and suggest potential screenings.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your AI Symptom Checker. Describe your symptoms (e.g., 'I have chest pain and fatigue') and I'll help you identify potential risks."}
        ]

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Type your symptoms here..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing symptoms..."):
                analysis = analyze_symptoms(user_input)
                bot_response = format_bot_response(analysis)
                st.markdown(bot_response)
                
                # Dynamic action buttons
                if analysis.get("possible_diseases") and diseases:
                    st.markdown("**Suggested Actions:**")
                    btn_cols = st.columns(len(analysis["possible_diseases"][:3]))
                    for i, d in enumerate(analysis["possible_diseases"][:3]):
                        with btn_cols[i]:
                            if st.button(f"🔍 Screen for {d['name']}", key=f"chat_btn_{i}"):
                                st.session_state.active_page = "🔬 Prediction Center"
                                st.info(f"Go to Prediction Center and select {d['name']}")
            
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    if st.sidebar.button("🗑️ Reset Consultation"):
        st.session_state.chat_history = []
        st.rerun()


# ─── PAGE 4: MODEL INSIGHTS ────────────────────────────────────
elif page == "📊 Model Insights":
    st.markdown("<h1 class='gradient-text'>Model Transparency & Performance</h1>", unsafe_allow_html=True)
    st.markdown("Explore the underlying AI models, their training metrics, and feature importance.")

    if not diseases:
        st.warning("No trained models available.")
        st.stop()

    # Overall Comparison
    st.subheader("System-Wide Performance")
    comp_df = pd.DataFrame(diseases)
    fig_comp = px.bar(comp_df.sort_values("accuracy"),
                     x="accuracy", y="display_name", orientation="h",
                     color="accuracy", color_continuous_scale="Plasma",
                     text=comp_df.sort_values("accuracy")["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
                     labels={"accuracy": "Accuracy", "display_name": "Condition"})
    
    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc"},
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    
    # Detail selection
    sel_disease_name = st.selectbox("Select Module for In-Depth Analysis", [d["display_name"] for d in diseases])
    sel_disease = next(d for d in diseases if d["display_name"] == sel_disease_name)
    
    try:
        _, _, meta = load_model(sel_disease["safe_name"])
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">Best Model</p><h3 style="margin:0;">{meta['best_model']}</h3></div>""", unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">Precision (Acc)</p><h3 style="margin:0;">{meta['accuracy']*100:.1f}%</h3></div>""", unsafe_allow_html=True)
        with m_c3:
            st.markdown(f"""<div class="glass-card"><p style="color:#94a3b8; margin:0;">ROC-AUC</p><h3 style="margin:0;">{meta['roc_auc']}</h3></div>""", unsafe_allow_html=True)

        i_col1, i_col2 = st.columns(2)
        
        with i_col1:
            st.subheader("Confusion Matrix")
            cm = np.array(meta["confusion_matrix"])
            # Normalize for better visualization if multi-class
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Viridis",
                labels=dict(x="Predicted", y="Actual"),
                x=list(meta.get("label_map", {}).values()) if meta.get("label_map") else ["Healthy", "Diseased"],
                y=list(meta.get("label_map", {}).values()) if meta.get("label_map") else ["Healthy", "Diseased"]
            )
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc"})
            st.plotly_chart(fig_cm, use_container_width=True)

        with i_col2:
            st.subheader("Feature Importance")
            if meta.get("feature_importance"):
                fi_df = pd.DataFrame(list(meta["feature_importance"].items()), columns=["Feature", "Score"]).head(10)
                fig_fi = px.bar(fi_df.sort_values("Score"), x="Score", y="Feature", orientation="h",
                               color="Score", color_continuous_scale="Magma")
                fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc"})
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance data not available for this model architecture.")

    except Exception as e:
        st.error(f"Error loading metadata: {e}")

# ─── FOOTER ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>MediScan AI v2.0 • Powered by Advanced Machine Learning • Designed for Healthcare Excellence</p>", unsafe_allow_html=True)
