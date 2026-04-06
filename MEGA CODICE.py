import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import factorial
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GOTHIC ORACLE v11.0 PRO", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    .stApp { background-color: #030303; color: #d0d0d0; }

    .gothic-title {
        font-family: 'UnifrakturMaguntia', cursive;
        color: #cc0000;
        text-align: center;
        font-size: 3.2rem;
        text-shadow: 0 0 30px #cc0000aa;
        margin-bottom: 0;
    }
    .version-tag {
        text-align: center;
        color: #666;
        font-size: 0.78rem;
        letter-spacing: 4px;
        margin-bottom: 20px;
        font-family: 'Share Tech Mono', monospace;
    }
    .section-header {
        font-family: 'Share Tech Mono', monospace;
        color: #cc0000;
        font-size: 1rem;
        letter-spacing: 3px;
        border-bottom: 1px solid #cc0000;
        padding-bottom: 4px;
        margin: 18px 0 12px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f0f0f, #1a0000);
        border: 1px solid #2a0000;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: #666;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 900;
        margin-top: 4px;
    }
    .kelly-box {
        background: linear-gradient(135deg, #001a00, #002a00);
        border: 1px solid #00aa00;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
        color: #00ff44;
        font-size: 1rem;
    }
    .warning-box {
        background-color: #1a1000;
        border-left: 3px solid #ff8800;
        padding: 8px 14px;
        border-radius: 3px;
        font-size: 0.85rem;
        margin: 3px 0;
    }
    .info-box {
        background-color: #000f1a;
        border-left: 3px solid #0088ff;
        padding: 8px 14px;
        border-radius: 3px;
        font-size: 0.85rem;
        margin: 3px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1a0000;
        color: #ff4444;
        font-weight: 900;
        border: 2px solid #cc0000;
        font-size: 1rem;
        padding: 10px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# UTILS
# ============================================================

def clean(val, default=1.2):
    if pd.isna(val):
        return default
    if isinstance(val, str):
        val = val.replace(",", ".")
    try:
        v = float(val)
        return v if v > 0 else default
    except:
        return default


def clean_int(val, default=0):
    if pd.isna(val):
        return default
    try:
        return int(val)
    except:
        return default


def validate_partite(row):
    msgs = []
    for col in ["xG_Home", "xGA_Home", "xG_Away", "xGA_Away"]:
        v = clean(row.get(col, np.nan), default=-1)
        if v < 0:
            msgs.append(f"[{col}] valore non valido")
        elif v > 5.0:
            msgs.append(f"[{col}] = {v} sembra anomalo (>5.0)")
    for col in ["ELO_Home", "ELO_Away"]:
        v = clean(row.get(col, np.nan), default=-1)
        if v > 0 and (v < 800 or v > 2800):
            msgs.append(f"[{col}] = {v} fuori range (800-2800)")
    for col in ["Quota1", "QuotaX", "Quota2"]:
        v = clean(row.get(col, np.nan), default=-1)
        if v < 1.01 or v > 50:
            msgs.append(f"[{col}] = {v} fuori range (1.01-50)")
    return msgs


# ============================================================
# GLICKO-2
# ============================================================

GLICKO2_TAU = 0.5
GLICKO2_Q = np.log(10) / 400
GLICKO2_EPS = 1e-6


def glicko2_g(phi):
    return 1.0 / np.sqrt(1 + 3 * phi**2 / np.pi**2)


def glicko2_E(mu, mu_j, phi_j):
    return 1.0 / (1 + np.exp(-glicko2_g(phi_j) * (mu - mu_j)))


def glicko2_update(r, rd, sigma, opponents, scores, tau=GLICKO2_TAU):
    """
    r      : rating corrente (scala 1-3000)
    rd     : rating deviation corrente
    sigma  : volatilita corrente
    opponents: lista di (r_j, rd_j) avversari
    scores   : lista di risultati (1=vittoria, 0.5=pari, 0=sconfitta)
    Restituisce (r_new, rd_new, sigma_new)
    """
    if len(opponents) == 0:
        rd_new = min(np.sqrt(rd**2 + sigma**2), 350)
        return r, rd_new, sigma

    # Conversione in scala mu
    mu = (r - 1500) / 173.7178
    phi = rd / 173.7178

    mu_j_list = [(rj - 1500) / 173.7178 for rj, _ in opponents]
    phi_j_list = [rdj / 173.7178 for _, rdj in opponents]

    g_list = [glicko2_g(pj) for pj in phi_j_list]
    E_list = [glicko2_E(mu, mj, pj) for mj, pj in zip(mu_j_list, phi_j_list)]

    v_inv = sum(g**2 * E * (1 - E) for g, E in zip(g_list, E_list))
    v = 1.0 / v_inv if v_inv > 0 else 1e6

    delta = v * sum(g * (s - E) for g, E, s in zip(g_list, E_list, scores))

    # Volatilita (Illinois algorithm)
    a = np.log(sigma**2)
    phi2 = phi**2

    def f(x):
        ex = np.exp(x)
        num = ex * (delta**2 - phi2 - v - ex)
        den = 2 * (phi2 + v + ex)**2
        return num / den - (x - a) / tau**2

    A = a
    B = np.log(delta**2 - phi2 - v) if delta**2 > phi2 + v else a - tau
    fA, fB = f(A), f(B)
    while abs(B - A) > GLICKO2_EPS:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC

    sigma_new = np.exp(A / 2)
    phi_star = np.sqrt(phi2 + sigma_new**2)
    phi_new = 1.0 / np.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_new = mu + phi_new**2 * sum(g * (s - E) for g, E, s in zip(g_list, E_list, scores))

    r_new = 173.7178 * mu_new + 1500
    rd_new = 173.7178 * phi_new
    return r_new, rd_new, sigma_new


def compute_glicko2_ratings(storico_df):
    """
    Calcola i rating Glicko-2 da un dataframe storico con colonne:
    Data, Home, Away, Gol_Home, Gol_Away
    Restituisce dict: team -> (r, rd, sigma)
    """
    ratings = {}

    def get_rating(team):
        if team not in ratings:
            ratings[team] = (1500.0, 200.0, 0.06)
        return ratings[team]

    if storico_df is None or len(storico_df) == 0:
        return ratings

    df = storico_df.copy()
    if "Data" in df.columns:
        try:
            df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
            df = df.sort_values("Data")
        except:
            pass

    for _, row in df.iterrows():
        home = str(row.get("Home", "")).strip()
        away = str(row.get("Away", "")).strip()
        gh = clean_int(row.get("Gol_Home", 0))
        ga = clean_int(row.get("Gol_Away", 0))
        if not home or not away:
            continue

        rh, rdh, sh = get_rating(home)
        ra, rda, sa = get_rating(away)

        if gh > ga:
            sh_score, sa_score = 1.0, 0.0
        elif gh == ga:
            sh_score, sa_score = 0.5, 0.5
        else:
            sh_score, sa_score = 0.0, 1.0

        rh_new, rdh_new, sh_new = glicko2_update(rh, rdh, sh, [(ra, rda)], [sh_score])
        ra_new, rda_new, sa_new = glicko2_update(ra, rda, sa, [(rh, rdh)], [sa_score])

        ratings[home] = (rh_new, rdh_new, sh_new)
        ratings[away] = (ra_new, rda_new, sa_new)

    return ratings


# ============================================================
# ATTACK / DEFENSE STRENGTH (MLE iterativo con decay)
# ============================================================

DECAY_XI = 0.0065  # fattore decay: ~6.5 per 1000 giorni


def compute_attack_defense(storico_df, today=None):
    """
    Stima attack_i, defense_j, home_adv via MLE pesato con decay temporale.
    Restituisce dicts: attack{team}, defense{team}, home_adv (float)
    """
    if storico_df is None or len(storico_df) < 5:
        return {}, {}, 1.35

    df = storico_df.copy()
    if today is None:
        today = pd.Timestamp.today()

    if "Data" in df.columns:
        try:
            df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
        except:
            df["Data"] = today

    df["days_ago"] = (today - df["Data"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-DECAY_XI * df["days_ago"])

    teams = sorted(set(df["Home"].tolist() + df["Away"].tolist()))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Inizializzazione
    attack = {t: 1.0 for t in teams}
    defense = {t: 1.0 for t in teams}
    home_adv = 1.35

    rows = []
    for _, row in df.iterrows():
        h = str(row.get("Home", "")).strip()
        a = str(row.get("Away", "")).strip()
        gh = clean_int(row.get("Gol_Home", 0))
        ga = clean_int(row.get("Gol_Away", 0))
        w = float(row.get("weight", 1.0))
        if h in idx and a in idx:
            rows.append((h, a, gh, ga, w))

    if len(rows) < 5:
        return attack, defense, home_adv

    # Iterazione Dixon-Coles style
    for _ in range(100):
        avg_goals = np.mean([gh + ga for _, _, gh, ga, _ in rows]) / 2 + 1e-9

        new_attack = {t: 0.0 for t in teams}
        new_defense = {t: 0.0 for t in teams}
        w_total = {t: 0.0 for t in teams}

        for h, a, gh, ga, w in rows:
            lh = attack[h] * defense[a] * home_adv
            la = attack[a] * defense[h]
            lh = max(lh, 1e-6)
            la = max(la, 1e-6)

            new_attack[h] += w * gh / lh
            new_attack[a] += w * ga / la
            new_defense[h] += w * ga / la
            new_defense[a] += w * gh / lh
            w_total[h] += w
            w_total[a] += w

        for t in teams:
            if w_total[t] > 0:
                attack[t] = new_attack[t] / w_total[t]
                defense[t] = new_defense[t] / w_total[t]

        # Normalizza
        mean_att = np.mean(list(attack.values()))
        mean_def = np.mean(list(defense.values()))
        if mean_att > 0:
            for t in teams:
                attack[t] /= mean_att
        if mean_def > 0:
            for t in teams:
                defense[t] /= mean_def

        # Home advantage
        lh_list = [attack[h] * defense[a] * home_adv for h, a, _, _, _ in rows]
        la_list = [attack[a] * defense[h] for h, a, _, _, _ in rows]
        num = sum(w * gh for _, _, gh, _, w in rows)
        den = sum(w * lh / home_adv for (h, a, gh, ga, w), lh in zip(rows, lh_list))
        if den > 0:
            home_adv = num / den

    return attack, defense, home_adv


# ============================================================
# DIXON-COLES CORRECTION
# ============================================================

def dc_tau(h, a, lh, la, rho=-0.13):
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    elif h == 0 and a == 1:
        return 1 + lh * rho
    elif h == 1 and a == 0:
        return 1 + la * rho
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0


# ============================================================
# BIVARIATE POISSON (Karlis-Ntzoufaris)
# ============================================================

def bivariate_poisson_pmf(h, a, l1, l2, l3):
    """
    P(H=h, A=a) nel modello Bivariate Poisson con lambda3 (covarianza).
    H ~ Poisson(l1+l3), A ~ Poisson(l2+l3), Cov(H,A)=l3
    """
    k_max = min(h, a)
    total = 0.0
    for k in range(k_max + 1):
        try:
            term = (poisson.pmf(h - k, l1) *
                    poisson.pmf(a - k, l2) *
                    poisson.pmf(k, l3) *
                    factorial(h) * factorial(a) / factorial(h - k) / factorial(a - k))
            total += term
        except:
            pass
    return total


def compute_prob_models(lh, la, l3=0.05, rho=-0.13, max_goals=8):
    """
    Calcola probabilita da tre modelli:
    - Poisson base
    - Dixon-Coles
    - Bivariate Poisson
    Restituisce dict con p1, px, p2, p_goal, p_over25 per modello.
    """
    results = {}

    for model in ["poisson", "dc", "bvp"]:
        p1, px, p2 = 0.0, 0.0, 0.0
        p_goal, p_over25 = 0.0, 0.0
        total = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if model == "poisson":
                    p = poisson.pmf(h, lh) * poisson.pmf(a, la)
                elif model == "dc":
                    p = poisson.pmf(h, lh) * poisson.pmf(a, la) * dc_tau(h, a, lh, la, rho)
                else:  # bvp
                    p = bivariate_poisson_pmf(h, a, max(lh - l3, 1e-6), max(la - l3, 1e-6), l3)

                total += p
                if h > a:
                    p1 += p
                elif h == a:
                    px += p
                else:
                    p2 += p
                if h > 0 and a > 0:
                    p_goal += p
                if h + a > 2.5:
                    p_over25 += p

        if total > 0:
            results[model] = {
                "p1": p1 / total,
                "px": px / total,
                "p2": p2 / total,
                "p_goal": p_goal / total,
                "p_over25": p_over25 / total
            }
        else:
            results[model] = {"p1": 0.33, "px": 0.33, "p2": 0.34, "p_goal": 0.5, "p_over25": 0.5}

    return results


# ============================================================
# CALIBRAZIONE ISOTONICA
# ============================================================

def build_calibrators(previsioni_df):
    """
    Costruisce calibratori isotonic regression da storico previsioni.
    Input: df con colonne P1_pred, PX_pred, P2_pred, Esito (1/X/2)
    Restituisce dict con calibratori per P1, PX, P2.
    """
    calibrators = {"1": None, "X": None, "2": None}
    if previsioni_df is None or len(previsioni_df) < 10:
        return calibrators

    df = previsioni_df.copy()
    df["Esito"] = df["Esito"].astype(str).str.strip()

    for sign, col in [("1", "P1_pred"), ("X", "PX_pred"), ("2", "P2_pred")]:
        if col not in df.columns:
            continue
        probs = pd.to_numeric(df[col], errors="coerce").fillna(0.33).values
        outcomes = (df["Esito"] == sign).astype(float).values
        if len(np.unique(outcomes)) < 2:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, outcomes)
            calibrators[sign] = ir
        except:
            pass

    return calibrators


def calibrate_prob(p, calibrator):
    if calibrator is None:
        return p
    try:
        return float(calibrator.predict([p])[0])
    except:
        return p


# ============================================================
# BAYESIAN MODEL AVERAGING
# ============================================================

def bma_weights(previsioni_df):
    """
    Calcola pesi BMA per i tre modelli (poisson, dc, bvp)
    usando log-score su storico previsioni se disponibile.
    Colonne richieste: P1_poisson, P1_dc, P1_bvp, ... , Esito
    Se non disponibili usa pesi uniformi.
    """
    default = {"poisson": 1/3, "dc": 1/3, "bvp": 1/3}
    if previsioni_df is None or len(previsioni_df) < 10:
        return default

    df = previsioni_df.copy()
    cols_needed = ["P1_poisson", "P1_dc", "P1_bvp",
                   "PX_poisson", "PX_dc", "PX_bvp",
                   "P2_poisson", "P2_dc", "P2_bvp", "Esito"]
    if not all(c in df.columns for c in cols_needed):
        return default

    log_scores = {"poisson": 0.0, "dc": 0.0, "bvp": 0.0}
    for _, row in df.iterrows():
        esito = str(row["Esito"]).strip()
        for m in ["poisson", "dc", "bvp"]:
            if esito == "1":
                p = clean(row.get(f"P1_{m}", 0.33))
            elif esito == "X":
                p = clean(row.get(f"PX_{m}", 0.33))
            else:
                p = clean(row.get(f"P2_{m}", 0.33))
            log_scores[m] += np.log(max(p, 1e-9))

    # Softmax sui log-score
    vals = np.array(list(log_scores.values()))
    vals -= vals.max()
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    models = list(log_scores.keys())
    weights = {m: exp_vals[i] / total for i, m in enumerate(models)}
    return weights


def ensemble_probs(model_results, weights):
    """
    Combina i risultati dei modelli con pesi BMA.
    """
    keys = ["p1", "px", "p2", "p_goal", "p_over25"]
    out = {k: 0.0 for k in keys}
    for m, w in weights.items():
        if m in model_results:
            for k in keys:
                out[k] += w * model_results[m][k]
    return out


# ============================================================
# ADAPTIVE KELLY
# ============================================================

def adaptive_kelly(p, q, rd=None, max_fraction=0.25, ruin_threshold=0.02):
    """
    Kelly frazionale adattivo.
    - rd: rating deviation Glicko-2 (incertezza). Piu alto = piu cauto.
    - ruin_threshold: frazione max del bankroll da rischiare in ogni bet
    - max_fraction: cap assoluto
    """
    if q <= 1.0 or p <= 0:
        return 0.0

    b = q - 1
    kelly_full = (p * b - (1 - p)) / b

    if kelly_full <= 0:
        return 0.0

    # Fattore di incertezza da Glicko-2 RD
    # RD standard ~ 50 (molto certo) a 350 (nuovo team)
    # Piu alta la RD, piu riduciamo la frazione
    if rd is not None:
        uncertainty_factor = max(0.1, 1 - (rd - 50) / 600)
    else:
        uncertainty_factor = 0.5  # half-Kelly di default

    kelly_adj = kelly_full * uncertainty_factor

    # Vincolo di rovina
    kelly_adj = min(kelly_adj, ruin_threshold / (1 - ruin_threshold))

    # Cap assoluto
    kelly_adj = min(kelly_adj, max_fraction)

    return max(0.0, kelly_adj)


# ============================================================
# MARKOV STABILITY (ricalibrato con Glicko-2 RD)
# ============================================================

def markov_stability(eh, ea, rd_h, rd_a, xh, xa, xah, xaa):
    """
    Stabilita che tiene conto della RD Glicko-2.
    Piu bassa la RD (rating certo), piu alta la stabilita.
    """
    # Dominance da differenza rating (normalizzata)
    r_diff = abs(eh - ea)
    prob_dominance = min(r_diff / 600.0, 1.0)

    # Penalita per incertezza RD
    avg_rd = (rd_h + rd_a) / 2
    rd_penalty = max(0.0, 1.0 - (avg_rd - 50) / 500)

    # Efficienza xG
    total_xg = xh + xa + xah + xaa + 0.1
    scoring_eff = (xh + xa) / total_xg

    markov = (prob_dominance * 0.5 + scoring_eff * 0.3 + rd_penalty * 0.2)
    return markov


# ============================================================
# ENGINE COMPLETO PER UNA PARTITA
# ============================================================

def analyze_match(row, attack, defense, home_adv,
                  glicko_ratings, calibrators, bma_w,
                  l3=0.05):
    try:
        home = str(row.get("Home", "?")).strip()
        away = str(row.get("Away", "?")).strip()

        xh  = clean(row.get("xG_Home",  1.2))
        xah = clean(row.get("xGA_Home", 1.2))
        xa  = clean(row.get("xG_Away",  1.2))
        xaa = clean(row.get("xGA_Away", 1.2))
        q1  = clean(row.get("Quota1",   2.0))
        qX  = clean(row.get("QuotaX",   3.3))
        q2  = clean(row.get("Quota2",   3.5))

        # Assenti (penalita xG)
        assenti_h = clean_int(row.get("Assenti_Home", 0))
        assenti_a = clean_int(row.get("Assenti_Away", 0))
        xh  *= max(0.7, 1 - 0.04 * assenti_h)
        xah *= max(0.7, 1 - 0.03 * assenti_h)
        xa  *= max(0.7, 1 - 0.04 * assenti_a)
        xaa *= max(0.7, 1 - 0.03 * assenti_a)

        # Forma recente (opzionale: punti ultimi 5, scala 0-15)
        form_h = clean(row.get("Form_Home", 7.5), default=7.5)
        form_a = clean(row.get("Form_Away", 7.5), default=7.5)
        form_factor_h = 0.85 + 0.15 * (form_h / 15.0)
        form_factor_a = 0.85 + 0.15 * (form_a / 15.0)

        # Glicko-2
        r_h, rd_h, _ = glicko_ratings.get(home, (1500.0, 200.0, 0.06))
        r_a, rd_a, _ = glicko_ratings.get(away, (1500.0, 200.0, 0.06))

        # Override ELO con Glicko se non fornito
        elo_h = clean(row.get("ELO_Home", r_h), default=r_h)
        elo_a = clean(row.get("ELO_Away", r_a), default=r_a)

        # Lambda base da ELO
        elo_diff = (elo_h - elo_a) / 400.0
        elo_factor_h = 1.2 ** elo_diff
        elo_factor_a = 1.2 ** (-elo_diff)

        # Lambda da attack/defense strength storico
        att_h = attack.get(home, 1.0)
        att_a = attack.get(away, 1.0)
        def_h = defense.get(home, 1.0)
        def_a = defense.get(away, 1.0)

        # Lambda composita: xG + strength + elo + forma
        has_strength = len(attack) > 0
        if has_strength:
            lh_base = ((xh + xaa) / 2) * att_h * def_a * home_adv * form_factor_h
            la_base = ((xa + xah) / 2) * att_a * def_h * form_factor_a
        else:
            lh_base = ((xh + xaa) / 2) * elo_factor_h * form_factor_h
            la_base = ((xa + xah) / 2) * elo_factor_a * form_factor_a

        lh = np.clip(lh_base, 0.05, 8.0)
        la = np.clip(la_base, 0.05, 8.0)

        # Modelli
        model_results = compute_prob_models(lh, la, l3=l3)

        # Ensemble BMA
        ens = ensemble_probs(model_results, bma_w)
        p1_raw = ens["p1"]
        px_raw = ens["px"]
        p2_raw = ens["p2"]

        # Calibrazione isotonica
        p1 = calibrate_prob(p1_raw, calibrators["1"])
        px = calibrate_prob(px_raw, calibrators["X"])
        p2 = calibrate_prob(p2_raw, calibrators["2"])

        # Ri-normalizza dopo calibrazione
        tot = p1 + px + p2
        if tot > 0:
            p1 /= tot
            px /= tot
            p2 /= tot

        p_goal   = ens["p_goal"]
        p_over25 = ens["p_over25"]

        # Best sign
        candidates = [(p1, q1, "1"), (px, qX, "X"), (p2, q2, "2")]
        best_p, best_q, best_sign = max(candidates, key=lambda x: x[0] * x[1])
        edge = best_p * best_q - 1

        # Kelly adattivo
        avg_rd = (rd_h + rd_a) / 2
        kelly = adaptive_kelly(best_p, best_q, rd=avg_rd)

        # Markov Stability
        stability = markov_stability(elo_h, elo_a, rd_h, rd_a, xh, xa, xah, xaa)

        # Probabilita per modello (per output dettagliato)
        detail = {
            "lh": round(lh, 3),
            "la": round(la, 3),
            "P1_poisson": round(model_results["poisson"]["p1"], 4),
            "PX_poisson": round(model_results["poisson"]["px"], 4),
            "P2_poisson": round(model_results["poisson"]["p2"], 4),
            "P1_dc":      round(model_results["dc"]["p1"], 4),
            "PX_dc":      round(model_results["dc"]["px"], 4),
            "P2_dc":      round(model_results["dc"]["p2"], 4),
            "P1_bvp":     round(model_results["bvp"]["p1"], 4),
            "PX_bvp":     round(model_results["bvp"]["px"], 4),
            "P2_bvp":     round(model_results["bvp"]["p2"], 4),
            "Glicko_H":   round(r_h, 1),
            "Glicko_A":   round(r_a, 1),
            "RD_H":       round(rd_h, 1),
            "RD_A":       round(rd_a, 1),
            "BMA_w_poi":  round(bma_w.get("poisson", 1/3), 3),
            "BMA_w_dc":   round(bma_w.get("dc", 1/3), 3),
            "BMA_w_bvp":  round(bma_w.get("bvp", 1/3), 3),
        }

        return {
            "P1": p1, "PX": px, "P2": p2,
            "P_GOAL": p_goal, "P_OVER25": p_over25,
            "BEST_SIGN": best_sign, "BOT_PROB": best_p,
            "EDGE": edge, "KELLY": kelly,
            "STABILITY": stability,
            "detail": detail
        }

    except Exception as e:
        return {
            "P1": 0.33, "PX": 0.33, "P2": 0.34,
            "P_GOAL": 0.5, "P_OVER25": 0.5,
            "BEST_SIGN": "N/A", "BOT_PROB": 0.33,
            "EDGE": 0.0, "KELLY": 0.0,
            "STABILITY": 0.3,
            "detail": {"error": str(e)}
        }


# ============================================================
# CONSIGLIO
# ============================================================

def give_advice(edge, stability, kelly):
    if edge > 0.05 and stability >= 0.40 and kelly > 0.01:
        return "CASSA FORTE 💰"
    if edge > 0.10 and stability < 0.40:
        return "SINGOLA FOLLE 🧨"
    if 0 < edge <= 0.05:
        return "VALUTARE 🔍"
    if edge <= 0:
        return "EVITA ❌"
    return "VALUTARE 🔍"


# ============================================================
# COLORI CELLA
# ============================================================

def color_cell(val, col_name):
    s = str(val)
    pos = ["GOAL", "OVER", "ALTA", "CASSA", "FORTE"]
    neg = ["BASSA", "EVITA", "NO GOAL", "UNDER", "❌"]
    neu = ["MEDIA", "VALUTARE", "🔍", "SINGOLA", "🧨"]

    if col_name == "EDGE":
        if isinstance(val, (int, float)):
            return "background-color: #004d00; color: #00ff44" if val > 0.05 else "background-color: #4d0000; color: #ff6666"
    if col_name == "KELLY":
        if isinstance(val, (int, float)):
            if val > 0.05:
                return "background-color: #003300; color: #00ff44; font-weight: bold"
            elif val > 0:
                return "background-color: #001a00; color: #88ff88"
            return "background-color: #0d0000; color: #444"
    if any(k in s for k in neg):
        return "background-color: #4d0000; color: #ff9999; font-weight: bold"
    if any(k in s for k in neu):
        return "background-color: #2a1f00; color: #ffcc44; font-weight: bold"
    if any(k in s for k in pos):
        return "background-color: #004d00; color: #99ff99; font-weight: bold"
    return ""


# ============================================================
# UI PRINCIPALE
# ============================================================

st.markdown('<div class="gothic-title">Gothic Oracle v11.0 Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="version-tag">BVP · GLICKO-2 · BMA · DECAY · KELLY ADATTIVO · CALIBRAZIONE</div>', unsafe_allow_html=True)

# Sidebar: parametri avanzati
with st.sidebar:
    st.markdown("### ⚙️ Parametri Avanzati")
    rho_dc     = st.slider("Dixon-Coles rho", -0.30, 0.0, -0.13, 0.01,
                           help="Correlazione DC. Valori piu negativi aumentano P(0-0).")
    l3_bvp     = st.slider("BVP lambda3 (covarianza)", 0.01, 0.30, 0.05, 0.01,
                           help="Covarianza Bivariate Poisson. Piu alto = piu correlazione gol.")
    decay_xi   = st.slider("Decay temporale xi", 0.001, 0.015, 0.0065, 0.0005,
                           help="Velocita decadimento dati storici. 0.0065 = ~50% peso dopo 107gg.")
    ruin_thr   = st.slider("Soglia rovina Kelly", 0.01, 0.10, 0.02, 0.01,
                           help="Percentuale massima bankroll per singola scommessa.")
    st.markdown("---")
    st.markdown("### 📋 Struttura Excel Richiesta")
    st.markdown("""
    **Sheet: Partite** *(obbligatorio)*
    Home, Away, xG_Home, xGA_Home, ELO_Home,
    xG_Away, xGA_Away, ELO_Away,
    Quota1, QuotaX, Quota2,
    Assenti_Home, Assenti_Away,
    Form_Home, Form_Away

    **Sheet: Storico** *(consigliato)*
    Data, Home, Away, Gol_Home, Gol_Away,
    xG_Home, xG_Away

    **Sheet: Previsioni** *(per calibrazione)*
    Data, Home, Away,
    P1_pred, PX_pred, P2_pred,
    Esito (1/X/2)
    """)

# Upload
uploaded_file = st.file_uploader(
    "Carica Excel con i fogli: Partite, Storico (opz.), Previsioni (opz.)",
    type="xlsx"
)

if uploaded_file:
    # Lettura sheets
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names

    # Partite (obbligatorio)
    sheet_partite = next((s for s in sheet_names if "artit" in s.lower()), sheet_names[0])
    df_partite = xl.parse(sheet_partite)

    # Storico (opzionale)
    sheet_storico = next((s for s in sheet_names if "toric" in s.lower()), None)
    df_storico = xl.parse(sheet_storico) if sheet_storico else None

    # Previsioni (opzionale)
    sheet_prev = next((s for s in sheet_names if "revis" in s.lower()), None)
    df_prev = xl.parse(sheet_prev) if sheet_prev else None

    # Info sheets caricati
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(f'<div class="info-box">✅ <b>Partite</b>: {len(df_partite)} righe</div>', unsafe_allow_html=True)
    with col_info2:
        n_stor = len(df_storico) if df_storico is not None else 0
        color = "info-box" if n_stor >= 5 else "warning-box"
        st.markdown(f'<div class="{color}">{"✅" if n_stor >= 5 else "⚠️"} <b>Storico</b>: {n_stor} partite</div>', unsafe_allow_html=True)
    with col_info3:
        n_prev = len(df_prev) if df_prev is not None else 0
        color = "info-box" if n_prev >= 10 else "warning-box"
        st.markdown(f'<div class="{color}">{"✅" if n_prev >= 10 else "⚠️"} <b>Previsioni</b>: {n_prev} righe (min 10 per calibrazione)</div>', unsafe_allow_html=True)

    # Pulizia numerica Partite
    num_cols = ["xG_Home", "xGA_Home", "ELO_Home", "xG_Away", "xGA_Away", "ELO_Away",
                "Quota1", "QuotaX", "Quota2", "Assenti_Home", "Assenti_Away",
                "Form_Home", "Form_Away"]
    for col in num_cols:
        if col in df_partite.columns and df_partite[col].dtype == object:
            df_partite[col] = df_partite[col].astype(str).str.replace(",", ".").astype(float, errors="ignore")

    # Validazione input
    all_warnings = []
    for i, row in df_partite.iterrows():
        w = validate_partite(row)
        if w:
            h = row.get("Home", f"Riga {i}")
            a = row.get("Away", "")
            for msg in w:
                all_warnings.append(f"<b>{h} vs {a}</b> - {msg}")

    if all_warnings:
        with st.expander(f"⚠️ {len(all_warnings)} avvisi sui dati di input"):
            for w in all_warnings:
                st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    if st.button("⚡ ESEGUI ANALISI PROFESSIONALE"):
        with st.spinner("Calcolo modelli in corso..."):

            # 1. Glicko-2
            glicko_ratings = compute_glicko2_ratings(df_storico)

            # 2. Attack/Defense con decay
            DECAY_XI = decay_xi
            attack, defense, home_adv = compute_attack_defense(df_storico)

            # 3. Calibratori isotonic
            calibrators = build_calibrators(df_prev)
            calib_active = any(v is not None for v in calibrators.values())

            # 4. Pesi BMA
            bma_w = bma_weights(df_prev)

            # 5. Analisi partite
            results = []
            for _, row in df_partite.iterrows():
                res = analyze_match(
                    row, attack, defense, home_adv,
                    glicko_ratings, calibrators, bma_w,
                    l3=l3_bvp
                )
                results.append(res)

        # Costruzione df risultati
        df_out = df_partite.copy().reset_index(drop=True)
        for key in ["P1", "PX", "P2", "P_GOAL", "P_OVER25",
                    "BEST_SIGN", "BOT_PROB", "EDGE", "KELLY", "STABILITY"]:
            df_out[key] = [r[key] for r in results]

        df_out["GOAL/NOGOAL"] = df_out["P_GOAL"].apply(lambda x: "GOAL" if x > 0.52 else "NO GOAL")
        df_out["U/O 2.5"]     = df_out["P_OVER25"].apply(lambda x: "OVER 2.5" if x > 0.55 else "UNDER 2.5")
        df_out["STABILITA"]   = df_out["STABILITY"].apply(
            lambda x: "ALTA" if x >= 0.50 else ("MEDIA" if x >= 0.30 else "BASSA")
        )
        df_out["CONSIGLIO"] = df_out.apply(
            lambda r: give_advice(r["EDGE"], r["STABILITY"], r["KELLY"]), axis=1
        )

        # ── METRICHE RIEPILOGATIVE ─────────────────────────────────────
        st.markdown('<div class="section-header">RIEPILOGO SESSIONE</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        counts = df_out["CONSIGLIO"].value_counts()
        items = [
            ("CASSA FORTE 💰",  "#00ff44"),
            ("SINGOLA FOLLE 🧨","#ff8800"),
            ("VALUTARE 🔍",     "#aaaaaa"),
            ("EVITA ❌",        "#ff4444"),
        ]
        for col, (label, color) in zip([c1, c2, c3, c4], items):
            n = counts.get(label, 0)
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{n}</div></div>',
                unsafe_allow_html=True
            )
        edge_medio = df_out["EDGE"].mean()
        c5.markdown(
            f'<div class="metric-card"><div class="label">EDGE MEDIO</div>'
            f'<div class="value" style="color:#ff4444">{edge_medio:.1%}</div></div>',
            unsafe_allow_html=True
        )

        # Info modelli attivi
        st.markdown('<div class="section-header">MODELLI ATTIVI</div>', unsafe_allow_html=True)
        mod_cols = st.columns(4)
        mod_info = [
            ("Glicko-2",    f"{len(glicko_ratings)} squadre calibrate" if glicko_ratings else "Non disponibile (inserisci Storico)"),
            ("Strength MLE",f"Home adv: {home_adv:.2f}" if attack else "Non disponibile (inserisci Storico)"),
            ("Calibrazione",f"Attiva su {sum(1 for v in calibrators.values() if v)} segni" if calib_active else "Non attiva (inserisci Previsioni, min 10)"),
            ("BMA Weights", f"Poi:{bma_w['poisson']:.2f} DC:{bma_w['dc']:.2f} BVP:{bma_w['bvp']:.2f}"),
        ]
        for col, (label, val) in zip(mod_cols, mod_info):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div style="font-size:0.8rem; color:#aaa; margin-top:6px">{val}</div></div>',
                unsafe_allow_html=True
            )

        # ── TABELLA PRINCIPALE ─────────────────────────────────────────
        st.markdown('<div class="section-header">ANALISI PARTITE</div>', unsafe_allow_html=True)

        view_cols = ["Home", "Away", "BEST_SIGN", "BOT_PROB", "EDGE", "KELLY",
                     "CONSIGLIO", "GOAL/NOGOAL", "U/O 2.5", "STABILITA", "STABILITY"]
        view_cols = [c for c in view_cols if c in df_out.columns]

        styled = (
            df_out[view_cols].style
            .format({
                "BOT_PROB": "{:.1%}",
                "EDGE":     "{:.1%}",
                "KELLY":    "{:.1%}",
                "STABILITY":"{:.1%}",
            })
            .map(lambda x: color_cell(x, "EDGE"),   subset=["EDGE"])
            .map(lambda x: color_cell(x, "KELLY"),  subset=["KELLY"])
            .map(lambda x: color_cell(x, "OTHER"),
                 subset=["GOAL/NOGOAL", "U/O 2.5", "STABILITA", "BEST_SIGN", "CONSIGLIO"])
        )
        st.dataframe(styled, use_container_width=True)

        # ── DETTAGLIO PER PARTITA ──────────────────────────────────────
        st.markdown('<div class="section-header">DETTAGLIO PER PARTITA</div>', unsafe_allow_html=True)

        for i, (_, row) in enumerate(df_out.iterrows()):
            home_n = row.get("Home", "?")
            away_n = row.get("Away", "?")
            consiglio = row.get("CONSIGLIO", "")
            det = results[i].get("detail", {})

            with st.expander(f"⚔️  {home_n}  vs  {away_n}  -  {consiglio}"):

                # Probabilita per modello
                st.markdown("**Confronto modelli:**")
                mc1, mc2, mc3 = st.columns(3)
                for col, model in zip([mc1, mc2, mc3], ["poisson", "dc", "bvp"]):
                    labels = {"poisson": "Poisson Base", "dc": "Dixon-Coles", "bvp": "Bivariate Poisson"}
                    col.markdown(f"*{labels[model]}*")
                    col.metric("P(1)", f"{det.get(f'P1_{model}', 0):.1%}")
                    col.metric("P(X)", f"{det.get(f'PX_{model}', 0):.1%}")
                    col.metric("P(2)", f"{det.get(f'P2_{model}', 0):.1%}")

                st.markdown("---")

                # Probabilita finali calibrate
                dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                dc1.metric("P(1) cal.", f"{row['P1']:.1%}")
                dc2.metric("P(X) cal.", f"{row['PX']:.1%}")
                dc3.metric("P(2) cal.", f"{row['P2']:.1%}")
                dc4.metric("GOAL",      f"{row['P_GOAL']:.1%}")
                dc5.metric("OVER 2.5",  f"{row['P_OVER25']:.1%}")

                st.markdown("---")

                # Glicko-2 e lambda
                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric(f"Glicko {home_n}", f"{det.get('Glicko_H', 1500):.0f}", f"RD: {det.get('RD_H', 200):.0f}")
                gc2.metric(f"Glicko {away_n}", f"{det.get('Glicko_A', 1500):.0f}", f"RD: {det.get('RD_A', 200):.0f}")
                gc3.metric("Lambda Home", f"{det.get('lh', 0):.3f}")
                gc4.metric("Lambda Away", f"{det.get('la', 0):.3f}")

                st.markdown("---")

                # BMA weights
                st.markdown(
                    f"**Pesi BMA:** Poisson `{det.get('BMA_w_poi', 0.33):.2f}` "
                    f"| Dixon-Coles `{det.get('BMA_w_dc', 0.33):.2f}` "
                    f"| BVP `{det.get('BMA_w_bvp', 0.33):.2f}`"
                )

                # Kelly box
                if row["KELLY"] > 0:
                    q_map = {"1": "Quota1", "X": "QuotaX", "2": "Quota2"}
                    q_val = clean(row.get(q_map.get(str(row["BEST_SIGN"]), "Quota1"), 2.0))
                    st.markdown(
                        f'<div class="kelly-box">'
                        f'💼 Kelly adattivo: punta il <b>{row["KELLY"]:.1%}</b> del bankroll '
                        f'su <b>{row["BEST_SIGN"]}</b> @ {q_val:.2f} '
                        f'| Edge: {row["EDGE"]:.1%}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Kelly = 0% - nessun valore atteso positivo. Non puntare.")

        st.success("✅ Analisi v11.0 Pro completata.")

