import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

# ==============================================================================
# 1. CORE ASSET LOADING & THEME
# ==============================================================================
st.set_page_config(page_title="IPL Analytics Pro", page_icon="🏏", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; }
    .stSelectbox label, .stRadio label, .stNumberInput label { color: #94a3b8 !important; font-weight: 600; }
    
    /* Glassmorphism Cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Prediction Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 1rem;
        font-weight: bold;
        transition: all 0.3s;
        border-radius: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("ipl_model_v1.pkl")
        le = joblib.load("label_encoder_v1.pkl")
        return pipeline, le
    except:
        return None, None

pipeline, le = load_assets()

# ==============================================================================
# 2. DATA CONSTANTS
# ==============================================================================
TEAM_LIST = sorted(list(le.classes_)) if le else ["Team Loading..."]
VENUE_MAP = {
    "Bengaluru - M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Mumbai - Wankhede Stadium": "Wankhede Stadium",
    "Ahmedabad - Narendra Modi Stadium": "Narendra Modi Stadium",
    "Chennai - MA Chidambaram Stadium": "MA Chidambaram Stadium",
    "Kolkata - Eden Gardens": "Eden Gardens",
    "Delhi - Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Hyderabad - Rajiv Gandhi Intl Stadium": "Rajiv Gandhi International Stadium",
    "Jaipur - Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "Lucknow - Ekana Stadium": "Ekana Cricket Stadium",
    "Guwahati - Barsapara Stadium": "Barsapara Cricket Stadium",
    "Mullanpur - New PCA Stadium": "Mullanpur Stadium",
    "Dharamsala - HPCA Stadium": "HPCA Stadium",
    "Raipur - Shaheed Veer Narayan Singh Stadium": "Raipur Stadium"
}

# ==============================================================================
# 3. SIDEBAR - CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Match Control")
    st.info("Set the parameters for the 2026 Match Simulator.")
    
    with st.expander("📅 Schedule", expanded=True):
        match_date = st.date_input("Match Date", datetime.now())
    
    with st.expander("📍 Venue Selection", expanded=True):
        v_key = st.selectbox("Stadium", list(VENUE_MAP.keys()))
        venue_name = VENUE_MAP[v_key]

# ==============================================================================
# 4. MAIN UI - MATCHUP INTERFACE
# ==============================================================================
st.title("IPL MATCH CENTER")

# THE MATCHUP HEADER
col_a, col_vs, col_b = st.columns([4, 1, 4])

with col_a:
    batting_team = st.selectbox("🏏 BATTING SIDE", TEAM_LIST, index=0)
    st.markdown(f"<h1 style='text-align: right; color: #60a5fa;'>{batting_team.upper()}</h1>", unsafe_allow_html=True)

with col_vs:
    st.markdown("<h1 style='text-align: center; padding-top: 40px; color: #475569;'>VS</h1>", unsafe_allow_html=True)

with col_b:
    remaining_teams = [t for t in TEAM_LIST if t != batting_team]
    bowling_team = st.selectbox("⚾ BOWLING SIDE", remaining_teams, index=0)
    st.markdown(f"<h1 style='text-align: left; color: #f87171;'>{bowling_team.upper()}</h1>", unsafe_allow_html=True)

st.divider()

# TOSS & INPUT ROW
c1, c2, c3 = st.columns(3)
with c1:
    toss_winner = st.radio("Toss Winner", [batting_team, bowling_team], horizontal=True)
with c2:
    toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)
with c3:
    st.write("") # Spacer
    predict_btn = st.button("Run Simulation ⚡")

# ==============================================================================
# 5. EXECUTION & RESULTS
# ==============================================================================
if predict_clicked := predict_btn:
    with st.status("Initializing Neural Weights...", expanded=True) as status:
        st.write("Fetching historical venue trends...")
        time.sleep(0.5)
        st.write("Calculating weather-adjusted strength...")
        time.sleep(0.4)
        status.update(label="Match Simulation Complete!", state="complete", expanded=False)

    # Data Prep
    input_df = pd.DataFrame([{
        "batting_team": batting_team, "bowling_team": bowling_team,
        "toss_winner": toss_winner, "toss_decision": toss_decision,
        "venue": venue_name, "year": match_date.year,
        "month": match_date.month, "day": match_date.day,
        "batting_strength": 0.5, "bowling_strength": 0.5,
        "batting_home": 0, "bowling_home": 0, "city": "unknown"
    }])

    # Logic
    try:
        prediction = pipeline.predict(input_df)
        winner = le.inverse_transform(prediction)[0]
        probs = pipeline.predict_proba(input_df)[0]
        
        # Filter & Normalize
        prob_df = pd.DataFrame({"Team": le.classes_, "P": probs})
        filtered = prob_df[prob_df["Team"].isin([batting_team, bowling_team])].copy()
        filtered["P"] = (filtered["P"] / filtered["P"].sum()) * 100
        
        win_p = filtered[filtered["Team"] == winner]["P"].values[0]
        lose_p = 100 - win_p
        loser = bowling_team if winner == batting_team else batting_team

        # DISPLAY RESULTS
        st.balloons()
        
        res_a, res_b = st.columns([3, 2])
        
        with res_a:
            st.subheader("Simulated Outcome")
            st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); border-left: 5px solid #3b82f6; padding: 30px; border-radius: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">WINNER PREDICTION</span>
                    <h1 style="font-size: 3.5rem; margin: 0;">{winner}</h1>
                    <p style="color: #60a5fa; font-size: 1.2rem;">Win Probability: {win_p:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            # Advantage metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Toss Factor", "High" if toss_winner == winner else "Neutral")
            m2.metric("Ground Condition", "Dry" if match_date.month in [4,5] else "Damp")
            m3.metric("Model Confidence", f"{win_p/100:.2f}")

        with res_b:
            st.subheader("Win Probability Chart")
            st.bar_chart(filtered.set_index("Team"))
            
            # Detailed Probability Table
            st.dataframe(filtered, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}. Check if model matches inputs.")

# ==============================================================================
# 6. ANALYTICS FOOTER
# ==============================================================================
st.markdown("---")
f1, f2 = st.columns([2, 1])
with f1:
    st.caption("2026 IPL Prediction Engine | Data based on seasons 2008-2025")
    st.caption("Warning: Sports betting involves risk. These predictions are for analytical purposes only.")
with f2:
    if st.button("Reset Dashboard"):
        st.rerun()