import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# ESTETICA GOTHIC
# ============================================================
st.set_page_config(page_title="GOTHIC ORACLE v15.0", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; }
    .gothic-title {
        font-family: 'UnifrakturMaguntia', cursive;
        color: #d10000; text-align: center; font-size: 3.5rem;
        text-shadow: 0 0 20px #d1000088;
    }
    .expert-tag {
        text-align: center; color: #888; font-size: 0.9rem;
        font-family: 'Share Tech Mono', monospace; margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CORE MATEMATICO: MARKOV + CONSIGLI ESPERTI
# ============================================================

def markov_engine_v15(lh, la, max_goals=6):
    """Simulazione a stati con influenza della dinamica di gioco"""
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    prob_matrix[0, 0] = 1.0
    dt = 1/90 
    
    for _ in range(90):
        new_matrix = np.zeros_like(prob_matrix)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if prob_matrix[i, j] > 0:
                    # Dinamica: Se una squadra è sotto di 2 gol, 'accelera' (lh/la aumentano)
                    # Se è sopra di 2, 'rallenta' (conservativa)
                    adj_lh = lh * (1.1 if j > i else 0.9 if i > j + 1 else 1.0)
                    adj_la = la * (1.1 if i > j else 0.9 if j > i + 1 else 1.0)
                    
                    ph, pa = adj_lh * dt, adj_la * dt
                    p_none = 1 - ph - pa
                    
                    if i < max_goals: new_matrix[i+1, j] += prob_matrix[i, j] * ph
                    if j < max_goals: new_matrix[i, j+1] += prob_matrix[i, j] * pa
                    new_matrix[i, j] += prob_matrix[i, j] * p_none
        prob_matrix = new_matrix

    p1 = np.sum(np.tril(prob_matrix, -1)) 
    p2 = np.sum(np.triu(prob_matrix, 1))
    px = np.trace(prob_matrix)
    return p1, px, p2

def analyze_match_expert(row, k_frac):
    try:
        # Recupero Dati
        h_name, a_name = row['Home'], row['Away']
        xh, xah = float(row['xG_Home_Field']), float(row['xGA_Home_Field'])
        xa, xaa = float(row['xG_Away_Field']), float(row['xGA_Away_Field'])
        eh, ea = float(row['ELO_Home']), float(row['ELO_Away'])
        q1, qx, q2 = float(row['Q1']), float(row['QX']), float(row['Q2'])
        inj_h, inj_a = float(row['Injuries_H']), float(row['Injuries_A'])

        # --- 1. INTEGRAZIONE CONSIGLIO ESPERTO: ATTACCO VS DIFESA ---
        # Bilanciamento pesato: diamo più importanza all'attacco della squadra in forma
        lh_base = (xh * 0.6 + xaa * 0.4) 
        la_base = (xa * 0.6 + xah * 0.4)

        # --- 2. INTEGRAZIONE CONSIGLIO ESPERTO: STRENGTH OF SCHEDULE (SOS) ---
        # Usiamo l'ELO per normalizzare: se l'avversario è forte, l'xG prodotto vale di più
        elo_diff = (eh - ea) / 400
        lh = lh_base * (1.1 ** elo_diff)
        la = la_base * (1.1 ** -elo_diff)

        # Penalità Infortuni
        lh *= (1 - (inj_h * 0.03))
        la *= (1 - (inj_a * 0.03))

        # --- 3. MOTORE MARKOVIANO ---
        p1, px, p2 = markov_engine_v15(max(0.1, lh), max(0.1, la))
        
        # Normalizzazione
        total = p1 + px + p2
        p1, px, p2 = p1/total, px/total, p2/total

        # Calcolo Valore (Edge)
        probs = {"1": p1, "X": px, "2": p2}
        quotas = {"1": q1, "X": qx, "2": q2}
        edges = {k: (probs[k] * quotas[k] - 1) for k in probs}
        best_s = max(edges, key=edges.get)
        
        # Kelly
        b = quotas[best_s] - 1
        kelly = max(0, ((probs[best_s] * b - (1 - probs[best_s])) / b) * k_frac) if b > 0 else 0

        # Under/Over (Poisson su Lambda esperti)
        p_over = 1 - poisson.cdf(2, lh + la)

        return {
            "Match": f"{h_name} vs {a_name}",
            "P1": p1, "PX": px, "P2": p2,
            "Edge %": edges[best_s],
            "Segno": best_s,
            "Kelly": kelly,
            "U/O 2.5": "OVER" if p_over > 0.54 else "UNDER",
            "SOS Index": "Alta Difficoltà" if abs(elo_diff) > 0.5 else "Equilibrata"
        }
    except Exception as e:
        return None

# ============================================================
# INTERFACCIA STREAMLIT
# ============================================================

st.markdown('<div class="gothic-title">Gothic Oracle v15.0</div>', unsafe_allow_html=True)
st.markdown('<div class="expert-tag">HYBRID ENGINE: ATTACK-DEFENSE BIAS & SOS NORMALIZATION</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Expert Settings")
    k_frac = st.slider("Frazione Kelly (Risk Management)", 0.05, 0.40, 0.20)
    st.info("Questa versione fonde la Catena di Markov con il bilanciamento pesato Attacco/Difesa suggerito dai tuoi esperti.")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame([{
        "Home": "Inter", "Away": "Napoli",
        "xG_Home_Field": 2.1, "xGA_Home_Field": 0.8,
        "xG_Away_Field": 1.4, "xGA_Away_Field": 1.2,
        "ELO_Home": 1900, "ELO_Away": 1820,
        "Rest_Days_H": 5, "Rest_Days_A": 3,
        "Injuries_H": 1, "Injuries_A": 3,
        "Q1": 1.80, "QX": 3.60, "Q2": 4.50
    }])

edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True)

if st.button("🔮 GENERA SENTENZA IBRIDA"):
    results = []
    for _, row in edited_df.dropna(subset=['Home']).iterrows():
        res = analyze_match_expert(row, k_frac)
        if res: results.append(res)
    
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.format({
            "P1": "{:.1%}", "PX": "{:.1%}", "P2": "{:.1%}",
            "Edge %": "{:.2%}", "Kelly": "{:.2%}"
        }).background_gradient(cmap='RdYlGn', subset=['Edge %']), use_container_width=True, hide_index=True)

