import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# ESTETICA GOTHIC SNIPER
# ============================================================
st.set_page_config(page_title="GOTHIC ORACLE v17.0", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; }
    .gothic-title {
        font-family: 'UnifrakturMaguntia', cursive;
        color: #d10000; text-align: center; font-size: 3.5rem;
        text-shadow: 0 0 30px #d10000;
    }
    .sniper-tag {
        text-align: center; color: #ff0000; font-size: 1rem;
        font-family: 'Share Tech Mono', monospace; margin-bottom: 30px;
        text-transform: uppercase; letter-spacing: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# MOTORE PREDITTIVO PURO (MARKOV SNIPER)
# ============================================================

def sniper_engine(lh, la, max_goals=7):
    """Motore Markov potenziato per la precisione del risultato finale"""
    # Matrice di transizione estesa per catturare punteggi alti
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    prob_matrix[0, 0] = 1.0
    
    # Simulazione dinamica 90 minuti
    for _ in range(90):
        new_matrix = np.zeros_like(prob_matrix)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if prob_matrix[i, j] > 0:
                    # Fattore 'Killer Instinct': chi è più forte tende ad accelerare se segna
                    # Fattore 'Capitulation': se una squadra subisce 3 gol, la sua difesa crolla
                    momentum_h = 1.10 if i > j else 1.0
                    momentum_a = 1.10 if j > i else 1.0
                    collapse_h = 1.20 if j >= 3 else 1.0
                    collapse_a = 1.20 if i >= 3 else 1.0
                    
                    ph = (lh / 90) * momentum_h * collapse_a
                    pa = (la / 90) * momentum_a * collapse_h
                    p_none = max(0, 1 - ph - pa)
                    
                    if i < max_goals: new_matrix[i+1, j] += prob_matrix[i, j] * ph
                    if j < max_goals: new_matrix[i, j+1] += prob_matrix[i, j] * pa
                    new_matrix[i, j] += prob_matrix[i, j] * p_none
        prob_matrix = new_matrix

    # Probabilità Segni
    p1 = np.sum(np.tril(prob_matrix, -1))
    p2 = np.sum(np.triu(prob_matrix, 1))
    px = np.trace(prob_matrix)
    
    # Calcolo Risultato Più Probabile (The Sniper Goal)
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    
    return p1, px, p2, f"{idx[0]}-{idx[1]}", prob_matrix

def analyze_sniper(row):
    try:
        # Dati Input
        h_name, a_name = row['Home'], row['Away']
        xh, xah = float(row['xG_Home_Field']), float(row['xGA_Home_Field'])
        xa, xaa = float(row['xG_Away_Field']), float(row['xGA_Away_Field'])
        eh, ea = float(row['ELO_Home']), float(row['ELO_Away'])
        
        # 1. BILANCIAMENTO PREDITTIVO (Anti-Rumore)
        # Calcoliamo la forza offensiva netta proiettata
        elo_diff = (eh - ea) / 400
        lh = ((xh + xaa) / 2) * (1.15 ** elo_diff)
        la = ((xa + xah) / 2) * (1.15 ** -elo_diff)
        
        # 2. ESECUZIONE SNIPER ENGINE
        p1, px, p2, exact, matrix = sniper_engine(max(0.1, lh), max(0.1, la))
        
        # 3. SENTENZA FINALE
        probs = {"1": p1, "X": px, "2": p2}
        predizione = max(probs, key=probs.get)
        
        # Probabilità Over 2.5 calcolata sulla matrice
        p_over = 1 - (matrix[0,0]+matrix[1,0]+matrix[0,1]+matrix[1,1]+matrix[2,0]+matrix[0,2]+matrix[2,1]+matrix[1,2])

        return {
            "MATCH": f"{h_name} vs {a_name}",
            "PROB. VITTORIA": probs[predizione],
            "SEGNO": predizione,
            "RISULTATO ESATTO": exact,
            "U/O 2.5": "OVER" if p_over > 0.52 else "UNDER",
            "POTENZA OFFENSIVA": "ALTA" if (lh+la) > 3.0 else "NORMALE"
        }
    except: return None

# ============================================================
# INTERFACCIA
# ============================================================

st.markdown('<div class="gothic-title">Gothic Oracle v17.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sniper-tag">Predictive Sniper Engine • Markov Exact Score</div>', unsafe_allow_html=True)

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame([{
        "Home": "Inter", "Away": "Milan",
        "xG_Home_Field": 1.13, "xGA_Home_Field": 1.4,
        "xG_Away_Field": 1.0, "xGA_Away_Field": 1.6,
        "ELO_Home": 1629, "ELO_Away": 1590,
        "Q1": 2.0, "QX": 3.2, "Q2": 4.1 # Le quote restano ma non influenzano la predizione
    }])

edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True)

if st.button("🎯 ATTIVA PROTOCOLLO SNIPER"):
    results = [analyze_sniper(row) for _, row in edited_df.dropna(subset=['Home']).iterrows()]
    if results:
        res_df = pd.DataFrame([r for r in results if r])
        st.markdown("### 🏹 RESPONSO PREDITTIVO")
        st.dataframe(res_df.style.format({"PROB. VITTORIA": "{:.1%}"}).highlight_max(axis=0, subset=['PROB. VITTORIA'], color='#4d0000'), use_container_width=True, hide_index=True)


