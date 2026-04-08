import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from scipy.special import iv # Funzione di Bessel per Skellam
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAZIONE ESTETICA GOTHIC
# ============================================================
st.set_page_config(page_title="GOTHIC ORACLE v13.0", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; }
    .gothic-title {
        font-family: 'UnifrakturMaguntia', cursive;
        color: #d10000; text-align: center; font-size: 3.5rem;
        text-shadow: 0 0 20px #d1000088; margin-bottom: -10px;
    }
    .version-tag {
        text-align: center; color: #666; font-size: 0.8rem;
        letter-spacing: 6px; margin-bottom: 25px; font-family: 'Share Tech Mono', monospace;
    }
    .metric-box {
        background: #0a0a0a; border: 1px solid #222; border-radius: 5px;
        padding: 15px; text-align: center;
    }
    .section-header {
        font-family: 'Share Tech Mono', monospace; color: #d10000;
        border-bottom: 1px solid #d10000; margin: 30px 0 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FUNZIONI MATEMATICHE AVANZATE
# ============================================================

def get_skellam_prob(lh, la, diff):
    """Calcola la probabilità esatta di una differenza reti usando Skellam"""
    return skellam.pmf(diff, lh, la)

def monte_carlo_simulation(lh, la, n_sim=10000):
    """Simula la partita 10.000 volte per calcolare la stabilità del pronostico"""
    home_goals = np.random.poisson(lh, n_sim)
    away_goals = np.random.poisson(la, n_sim)
    
    diff = home_goals - away_goals
    wins = np.sum(diff > 0) / n_sim
    draws = np.sum(diff == 0) / n_sim
    losses = np.sum(diff < 0) / n_sim
    
    # La stabilità è l'inverso della varianza dei risultati
    stability = 1.0 - (np.std(diff) / 5.0) 
    return wins, draws, losses, max(0.1, stability)

def calculate_kelly(prob, quota, fraction=0.20):
    """Kelly Frazionale (0.20 = 1/5 Kelly per massima sicurezza)"""
    if quota <= 1.0 or prob <= 0: return 0.0
    b = quota - 1
    k = (prob * b - (1 - prob)) / b
    return max(0.0, k * fraction)

# ============================================================
# ENGINE DI ANALISI
# ============================================================

def analyze_match_v13(row, rho, l3, k_frac):
    # Dati base
    h_name, a_name = row['Home'], row['Away']
    
    # xG specifici (Casa/Fuori) + ELO
    xh, xah = row['xG_Home_Field'], row['xGA_Home_Field']
    xa, xaa = row['xG_Away_Field'], row['xGA_Away_Field']
    elo_h, elo_a = row['ELO_Home'], row['ELO_Away']
    
    # Nuovi Dati: Fatica e Infortuni
    rest_h, rest_a = row['Rest_Days_H'], row['Rest_Days_A']
    inj_h, inj_a = row['Injuries_H'], row['Injuries_A'] # Scala 0-10
    
    # --- CALCOLO LAMBDA DINAMICO ---
    # 1. Base incrociata
    lh = (xh + xaa) / 2.0
    la = (xa + xah) / 2.0
    
    # 2. Correzione Fatica (se < 4 giorni di riposo, penalità)
    if rest_h < 4: lh *= 0.94
    if rest_a < 4: la *= 0.94
    
    # 3. Correzione Infortuni (Penalità 2% per ogni punto indice)
    lh *= (1 - (inj_h * 0.02))
    la *= (1 - (inj_a * 0.02))
    
    # 4. Shift ELO (Potere risolutivo)
    elo_diff = elo_h - elo_a
    lh *= (1 + (elo_diff / 1000.0))
    la *= (1 - (elo_diff / 1000.0))
    
    lh, la = max(0.1, lh), max(0.1, la)

    # --- SIMULAZIONE E MODELLI ---
    p1_mc, px_mc, p2_mc, stability = monte_carlo_simulation(lh, la)
    
    # Probabilità Pareggio corretta (Dixon-Coles style manuale)
    # Aggiungiamo una piccola correzione per i pareggi tipici dei campionati europei
    px_final = px_mc * (1 + abs(rho))
    
    # Probabilità Goal / Over 2.5
    p_goal = 1 - (poisson.pmf(0, lh) + poisson.pmf(0, la) - (poisson.pmf(0, lh) * poisson.pmf(0, la)))
    p_over = 1 - (poisson.pmf(0, lh)*poisson.pmf(0, la) + poisson.pmf(1, lh)*poisson.pmf(0, la) + 
                  poisson.pmf(0, lh)*poisson.pmf(1, la) + poisson.pmf(1, lh)*poisson.pmf(1, la) +
                  poisson.pmf(2, lh)*poisson.pmf(0, la) + poisson.pmf(0, lh)*poisson.pmf(2, la))

    # --- CALCOLO EDGE ---
    q1, qx, q2 = row['Q1'], row['QX'], row['Q2']
    probs = {"1": p1_mc, "X": px_final, "2": p2_mc}
    quotas = {"1": q1, "X": qx, "2": q2}
    
    # Normalizzazione
    total = sum(probs.values())
    for k in probs: probs[k] /= total
    
    edges = {k: (probs[k] * quotas[k] - 1) for k in probs}
    best_sign = max(edges, key=edges.get)
    
    kelly = calculate_kelly(probs[best_sign], quotas[best_sign], fraction=k_frac)

    return {
        "Match": f"{h_name} vs {a_name}",
        "P1": probs["1"], "PX": probs["X"], "P2": probs["2"],
        "Edge %": edges[best_sign],
        "Segno": best_sign,
        "Quota": quotas[best_sign],
        "Kelly": kelly,
        "Stabilità": stability,
        "U/O 2.5": "OVER" if p_over > 0.53 else "UNDER",
        "G/NG": "GOAL" if p_goal > 0.52 else "NOGOAL"
    }

# ============================================================
# INTERFACCIA UTENTE
# ============================================================

st.markdown('<div class="gothic-title">Gothic Oracle v13.0</div>', unsafe_allow_html=True)
st.markdown('<div class="version-tag">SKELLAM · MONTE CARLO · FATIGUE ENGINE</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🛠️ Setup Matematico")
    rho = st.slider("Coefficiente Pareggio", -0.20, 0.0, -0.13)
    k_frac = st.slider("Frazione Kelly", 0.05, 0.40, 0.20)
    st.info("I dati 'Field' si riferiscono agli xG prodotti/subiti solo in Casa (per la Home) o solo in Trasferta (per l'Away).")

# Inizializzazione Tabella
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

if st.button("🔮 ANALIZZA CON POTENZA DI CALCOLO V13"):
    results = []
    df_valid = edited_df.dropna(subset=['Home', 'Q1'])
    
    for _, row in df_valid.iterrows():
        results.append(analyze_match_v13(row, rho, 0.05, k_frac))
    
    res_df = pd.DataFrame(results)
    
    st.markdown('<div class="section-header">SENTENZA DELL\'ORACOLO</div>', unsafe_allow_html=True)
    
    # Formattazione Colori
    def style_results(styler):
        styler.format({
            "P1": "{:.1%}", "PX": "{:.1%}", "P2": "{:.1%}",
            "Edge %": "{:.2%}", "Kelly": "{:.2%}", "Stabilità": "{:.1%}"
        })
        styler.background_gradient(cmap='RdYlGn', subset=['Edge %', 'Stabilità'])
        return styler

    st.dataframe(style_results(res_df.style), use_container_width=True, hide_index=True)
    
    # Box di riepilogo per la migliore giocata
    best_bet = res_df.loc[res_df['Edge %'].idxmax()]
    st.success(f"🎯 **MIGLIORE OPPORTUNITÀ:** {best_bet['Match']} -> Segno {best_bet['Segno']} (Edge: {best_bet['Edge %']:.2%})")


