"""
OFFSIDE — UCL Statistical Analyzer
Flask backend: UCL Round of 16 data pre-loaded — no upload needed.
Run: python app.py  →  open http://localhost:5000
"""

import io, base64, json, os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from flask import Flask, request, jsonify, render_template, session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

app = Flask(__name__)
app.secret_key = "offside_ucl_2026"

# ── Stored dataframe (in-memory per process) ──────────────────────────────────
_DF: pd.DataFrame | None = None
_SQUADS: list = []

FIVE_VARS  = ["Gls", "Ast", "xG", "Age", "Min"]
VAR_LABELS = {"Gls":"Goals","Ast":"Assists","xG":"Expected Goals (xG)","Age":"Age","Min":"Minutes Played"}

# ── Chart theme ───────────────────────────────────────────────────────────────
BG      = "#1e1e1e"
BG2     = "#2a2a2a"
GREEN   = "#00C853"
BLUE    = "#4FC3F7"
GRAY    = "#555555"
RED     = "#F44336"
GOLD    = "#FFD700"
TXT     = "#EEEEEE"
TXTS    = "#888888"
BORDER  = "#333333"

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TXT, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=TXTS)
    ax.set_ylabel(ylabel, fontsize=10, color=TXTS)
    ax.tick_params(colors=TXTS, labelsize=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color(BORDER)
    ax.yaxis.grid(True, alpha=0.15, color=BORDER, linestyle="--")
    ax.set_axisbelow(True)

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=BG, dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def _load_df(file_bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    keep = ["Player","Squad","Pos","Nation","Age","Min","Gls","Ast","xG","xAG","MP","Sh","SoT","CrdY","CrdR"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    if "Nation" in df.columns:
        df["Nation"] = df["Nation"].astype(str).str.split(" ").str[-1]
    df = df.dropna(subset=["Age","Min","Gls","Ast","xG"]).reset_index(drop=True)
    for col in ["Age","Min","Gls","Ast","xG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["PosSimple"] = df["Pos"].astype(str).str.split(",").str[0]
    return df

# ── Auto-load embedded Excel on startup ───────────────────────────────────────

def _auto_load():
    global _DF, _SQUADS
    here = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(here, "ucl_round_of_16.xlsx")
    if os.path.exists(xlsx_path):
        with open(xlsx_path, "rb") as f:
            _DF = _load_df(f.read())
        _SQUADS = sorted(_DF["Squad"].unique().tolist())

_auto_load()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    if _DF is not None:
        return jsonify({
            "loaded": True,
            "players": len(_DF),
            "clubs": _DF["Squad"].nunique(),
            "squads": _SQUADS,
        })
    return jsonify({"loaded": False})

@app.route("/api/upload", methods=["POST"])
def upload():
    global _DF, _SQUADS
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    try:
        _DF = _load_df(f.read())
        _SQUADS = sorted(_DF["Squad"].unique().tolist())
        return jsonify({
            "ok": True,
            "players": len(_DF),
            "clubs": _DF["Squad"].nunique(),
            "squads": _SQUADS,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _require_df():
    if _DF is None:
        return None, jsonify({"error": "No data loaded"}), 400
    return _DF, None, None

# ── HOME stats ────────────────────────────────────────────────────────────────

@app.route("/api/home")
def home_stats():
    df, err, code = _require_df()
    if err: return err, code

    # Top performers
    sub = df.copy()
    sub["G+A"] = sub["Gls"] + sub["Ast"]
    top = sub.nlargest(10, "G+A")[["Player","Squad","Gls","Ast","xG"]].reset_index(drop=True)

    # Club summary
    sq = df.groupby("Squad").agg(
        Players=("Player","count"),
        Goals=("Gls","sum"),
        Assists=("Ast","sum"),
        Avg_xG=("xG","mean"),
        Avg_Age=("Age","mean"),
    ).round(2).sort_values("Goals", ascending=False).reset_index()

    return jsonify({
        "kpis": {
            "players": len(df),
            "total_goals": int(df["Gls"].sum()),
            "total_assists": int(df["Ast"].sum()),
            "avg_xg": round(float(df["xG"].mean()), 3),
            "clubs": df["Squad"].nunique(),
            "top_scorer": df.loc[df["Gls"].idxmax(), "Player"],
            "top_scorer_goals": int(df["Gls"].max()),
        },
        "top_players": top.to_dict("records"),
        "squad_summary": sq.to_dict("records"),
    })

# ── Chart: top players ─────────────────────────────────────────────────────────

@app.route("/api/chart/top_players")
def chart_top_players():
    df, err, code = _require_df()
    if err: return err, code
    sub = df.copy(); sub["G+A"] = sub["Gls"] + sub["Ast"]
    top = sub.nlargest(10, "G+A").reset_index(drop=True)
    names = [p[:15] for p in top["Player"].values]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.bar(x - w/2, top["Gls"].values, width=w, color=GREEN, alpha=0.9, label="Goals", zorder=3)
    ax.bar(x + w/2, top["Ast"].values, width=w, color=BLUE, alpha=0.85, label="Assists", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9, color=TXTS)
    ax.legend(fontsize=10, framealpha=0.15, labelcolor=TXT)
    _style(ax, "Top 10 Players — Goals & Assists", "Player", "Count")
    fig.tight_layout(pad=1.5)
    return jsonify({"img": _fig_to_b64(fig)})

# ── ANALYTICS ─────────────────────────────────────────────────────────────────

@app.route("/api/analytics/descriptive")
def descriptive():
    df, err, code = _require_df()
    if err: return err, code
    rows = []
    for var in FIVE_VARS:
        s = df[var].dropna()
        rows.append({
            "Variable": VAR_LABELS[var], "Count": int(s.count()),
            "Mean": round(float(s.mean()),3), "Median": round(float(s.median()),3),
            "Std Dev": round(float(s.std()),3), "Variance": round(float(s.var()),3),
            "Min": round(float(s.min()),3), "Max": round(float(s.max()),3),
            "Skewness": round(float(s.skew()),3), "Kurtosis": round(float(s.kurt()),3),
        })
    return jsonify(rows)

@app.route("/api/analytics/ci")
def confidence_intervals():
    df, err, code = _require_df()
    if err: return err, code
    rows = []
    for var in FIVE_VARS:
        s = df[var].dropna(); n = len(s); mean = s.mean(); se = scipy_stats.sem(s)
        lo, hi = scipy_stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        rows.append({
            "Variable": VAR_LABELS[var], "n": n,
            "Mean": round(float(mean),3), "SE": round(float(se),4),
            "95% CI Lower": round(float(lo),3), "95% CI Upper": round(float(hi),3),
        })
    return jsonify(rows)

@app.route("/api/analytics/normality")
def normality():
    df, err, code = _require_df()
    if err: return err, code
    rows = []
    for var in FIVE_VARS:
        s = df[var].dropna()
        sample = s.sample(min(len(s), 5000), random_state=42)
        stat, p = scipy_stats.shapiro(sample)
        rows.append({
            "Variable": VAR_LABELS[var],
            "W Statistic": round(float(stat),4),
            "p-value": round(float(p),6),
            "Normal?": "✅ Yes" if p > 0.05 else "❌ No",
        })
    return jsonify(rows)

@app.route("/api/chart/histogram/<var>")
def chart_histogram(var):
    df, err, code = _require_df()
    if err: return err, code
    if var not in FIVE_VARS: return jsonify({"error":"bad var"}), 400
    data = df[var].dropna()
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.hist(data, bins=30, color=GREEN, alpha=0.7, edgecolor="none", density=True, zorder=3)
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 300)
    ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma), color=BLUE, linewidth=2,
            label=f"Normal  μ={mu:.2f}  σ={sigma:.2f}")
    ax.axvline(mu, color=GOLD, linestyle="--", linewidth=1.5, label=f"Mean = {mu:.2f}")
    ax.legend(fontsize=10, framealpha=0.15, labelcolor=TXT)
    _style(ax, f"Distribution of {VAR_LABELS[var]}", VAR_LABELS[var], "Density")
    fig.tight_layout(pad=1.5)
    return jsonify({"img": _fig_to_b64(fig)})

@app.route("/api/chart/boxplot/<var>")
def chart_boxplot(var):
    df, err, code = _require_df()
    if err: return err, code
    positions = df["PosSimple"].unique()
    groups = [df[df["PosSimple"] == p][var].dropna().values for p in positions]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color=GREEN, linewidth=2),
                    whiskerprops=dict(color=GRAY),
                    capprops=dict(color=GRAY),
                    flierprops=dict(marker="o", color=GRAY, markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(BG2); patch.set_edgecolor(GREEN); patch.set_alpha(0.9)
    ax.set_xticks(range(1, len(positions)+1))
    ax.set_xticklabels(positions, rotation=20, ha="right", fontsize=10, color=TXTS)
    _style(ax, f"{VAR_LABELS[var]} by Position", "Position", VAR_LABELS[var])
    fig.tight_layout(pad=1.5)
    return jsonify({"img": _fig_to_b64(fig)})

@app.route("/api/chart/poisson/<var>")
def chart_poisson(var):
    df, err, code = _require_df()
    if err: return err, code
    if var not in ["Gls","Ast"]: return jsonify({"error":"bad var"}), 400
    data = df[var].dropna().astype(int)
    lam = float(data.mean())
    x_vals = np.arange(0, min(int(data.max())+1, 20))
    obs = np.array([(data == x).sum() for x in x_vals])
    exp = scipy_stats.poisson.pmf(x_vals, lam) * len(data)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    w = 0.4
    ax.bar(x_vals - w/2, obs, width=w, color=GREEN, alpha=0.8, label="Observed", zorder=3)
    ax.bar(x_vals + w/2, exp, width=w, color=BLUE, alpha=0.8,
           label=f"Poisson  λ={lam:.2f}", zorder=3)
    ax.legend(fontsize=10, framealpha=0.15, labelcolor=TXT)
    ax.set_xlim(-0.5, len(x_vals)-0.5)
    _style(ax, f"Poisson Fit — {VAR_LABELS[var]}", VAR_LABELS[var], "Frequency")
    fig.tight_layout(pad=1.5)
    cdf_k5 = float(scipy_stats.poisson.cdf(5, lam))
    return jsonify({"img": _fig_to_b64(fig), "lambda": round(lam,3), "cdf_k5": round(cdf_k5,4)})

@app.route("/api/analytics/poisson_cdf")
def poisson_cdf():
    df, err, code = _require_df()
    if err: return err, code
    var = request.args.get("var","Gls")
    k = int(request.args.get("k", 5))
    lam = float(df[var].dropna().mean())
    p = float(scipy_stats.poisson.cdf(k, lam))
    return jsonify({"p": round(p,4), "lambda": round(lam,3), "k": k})

@app.route("/api/chart/heatmap")
def chart_heatmap():
    df, err, code = _require_df()
    if err: return err, code
    corr = df[FIVE_VARS].corr(method="pearson").round(3)
    labels = [VAR_LABELS[v] for v in FIVE_VARS]
    data = corr.values; n = len(labels)
    cmap = LinearSegmentedColormap.from_list("fm", [BLUE, BG2, GREEN], N=256)
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=TXTS)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = ["Goals","Assists","xG","Age","Minutes"]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10, color=TXTS)
    ax.set_yticklabels(short, fontsize=10, color=TXTS)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color=TXT)
    _style(ax, "Pearson Correlation Matrix")
    fig.tight_layout(pad=1.5)
    corr_data = {FIVE_VARS[i]: {FIVE_VARS[j]: round(float(data[i,j]),3)
                 for j in range(n)} for i in range(n)}
    return jsonify({"img": _fig_to_b64(fig), "corr": corr_data})

@app.route("/api/chart/goals_by_position")
def chart_goals_by_pos():
    df, err, code = _require_df()
    if err: return err, code
    sub = df.copy()
    sub["Gls_per90"] = sub["Gls"] / (sub["Min"]/90).replace(0, np.nan)
    pos_df = (sub.groupby("PosSimple")["Gls_per90"].mean().round(3)
              .sort_values(ascending=False).reset_index()
              .rename(columns={"PosSimple":"Position","Gls_per90":"Goals per 90 min"}))
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
    colors = [GREEN if p == "FW" else BG2 for p in pos_df["Position"].values]
    bars = ax.barh(pos_df["Position"].values, pos_df["Goals per 90 min"].values,
                   color=colors, edgecolor=GREEN, linewidth=0.8, height=0.55, zorder=3)
    for bar, val in zip(bars, pos_df["Goals per 90 min"].values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=TXTS)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.15, color=BORDER, linestyle="--")
    ax.yaxis.grid(False)
    _style(ax, "Avg Goals per 90 min by Position", "Goals / 90", "")
    fig.tight_layout(pad=1.5)
    return jsonify({"img": _fig_to_b64(fig), "data": pos_df.to_dict("records")})

# ── REGRESSION ────────────────────────────────────────────────────────────────

@app.route("/api/regression/simple")
def simple_reg():
    df, err, code = _require_df()
    if err: return err, code
    x_var = request.args.get("x","xG")
    y_var = request.args.get("y","Gls")
    if x_var not in FIVE_VARS or y_var not in FIVE_VARS or x_var == y_var:
        return jsonify({"error":"invalid vars"}), 400
    sub = df[[x_var, y_var]].dropna()
    x = sub[x_var].values; y = sub[y_var].values
    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    # scatter chart
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    ax.scatter(x, y, alpha=0.3, s=14, color=GRAY, edgecolors="none", zorder=3)
    xl = np.linspace(x.min(), x.max(), 300)
    ax.plot(xl, intercept + slope*xl, color=GREEN, linewidth=2.5, zorder=4,
            label=f"y = {slope:.4f}x + {intercept:.4f}   R²={r**2:.4f}")
    ax.legend(fontsize=10, framealpha=0.15, labelcolor=TXT)
    _style(ax, f"Regression: {VAR_LABELS[y_var]} ~ {VAR_LABELS[x_var]}",
           VAR_LABELS[x_var], VAR_LABELS[y_var])
    fig.tight_layout(pad=1.5)
    return jsonify({
        "img": _fig_to_b64(fig),
        "slope": round(float(slope),4), "intercept": round(float(intercept),4),
        "R2": round(float(r**2),4), "r": round(float(r),4),
        "p_value": round(float(p),6), "SE": round(float(se),4),
        "x_label": VAR_LABELS[x_var], "y_label": VAR_LABELS[y_var],
    })

@app.route("/api/regression/predict")
def reg_predict():
    df, err, code = _require_df()
    if err: return err, code
    x_var = request.args.get("x","xG")
    y_var = request.args.get("y","Gls")
    x_val = float(request.args.get("val",0))
    sub = df[[x_var,y_var]].dropna()
    slope, intercept, *_ = scipy_stats.linregress(sub[x_var].values, sub[y_var].values)
    y_hat = intercept + slope * x_val
    return jsonify({"y_hat": round(float(y_hat),3), "x_val": x_val,
                    "y_label": VAR_LABELS[y_var]})

@app.route("/api/regression/multiple")
def multiple_reg():
    df, err, code = _require_df()
    if err: return err, code
    sub = df[["Gls","xG","Min","Age"]].dropna()
    y = sub["Gls"].values
    X = np.column_stack([np.ones(len(sub)), sub["xG"].values,
                         sub["Min"].values, sub["Age"].values])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat)**2); ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    # residual chart
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.scatter(y_hat, y - y_hat, alpha=0.3, s=12, color=GRAY, edgecolors="none", zorder=3)
    ax.axhline(0, color=GREEN, linewidth=1.8, linestyle="--", zorder=4)
    _style(ax, "Residuals vs Fitted  (Multiple Regression)", "Fitted Values", "Residuals")
    fig.tight_layout(pad=1.5)
    return jsonify({
        "img": _fig_to_b64(fig),
        "intercept": round(float(coeffs[0]),4),
        "coef_xG": round(float(coeffs[1]),4),
        "coef_Min": round(float(coeffs[2]),6),
        "coef_Age": round(float(coeffs[3]),4),
        "R2": round(float(r2),4), "n": len(sub),
    })

# ── PREDICTOR ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    df, err, code = _require_df()
    if err: return err, code
    body = request.get_json()
    home, away = body.get("home",""), body.get("away","")
    if home == away or not home or not away:
        return jsonify({"error":"invalid teams"}), 400

    sq = df.groupby("Squad").agg(
        total_goals=("Gls","sum"), total_assists=("Ast","sum"),
        avg_xg=("xG","mean"), total_xg=("xG","sum"),
        avg_age=("Age","mean"), players=("Player","count"),
    ).reset_index()
    sq["gpp"] = sq["total_goals"] / sq["players"]
    sq["xg_acc"] = (sq["total_goals"] / sq["total_xg"].replace(0,1)).clip(0,2)

    def get(name):
        r = sq[sq["Squad"]==name]
        return r.iloc[0] if not r.empty else None

    h, a = get(home), get(away)
    if h is None or a is None:
        return jsonify({"error":"team not found"}), 404

    def score(r):
        return r["gpp"]*3.0 + r["avg_xg"]*2.5 + r["xg_acc"]*1.5 + (r["total_assists"]/max(r["players"],1))*1.0

    hs = score(h) * 1.08; as_ = score(a)
    tot = hs + as_
    rh, ra = hs/tot, as_/tot
    dp = (1 - abs(rh - ra)) * 0.25
    hp = rh*(1-dp); ap = ra*(1-dp)
    s = hp+dp+ap
    hp, dp, ap = hp/s*100, dp/s*100, ap/s*100

    if hp > ap+5: winner = home; wside = "home"
    elif ap > hp+5: winner = away; wside = "away"
    else: winner = "Too Close to Call"; wside = "draw"

    return jsonify({
        "home": home, "away": away,
        "home_p": round(hp,1), "draw_p": round(dp,1), "away_p": round(ap,1),
        "winner": winner, "wside": wside,
        "home_stats": {
            "goals": int(h["total_goals"]), "assists": int(h["total_assists"]),
            "avg_xg": round(float(h["avg_xg"]),3), "gpp": round(float(h["gpp"]),2),
            "xg_acc": round(float(h["xg_acc"]),2), "avg_age": round(float(h["avg_age"]),1),
            "players": int(h["players"]),
        },
        "away_stats": {
            "goals": int(a["total_goals"]), "assists": int(a["total_assists"]),
            "avg_xg": round(float(a["avg_xg"]),3), "gpp": round(float(a["gpp"]),2),
            "xg_acc": round(float(a["xg_acc"]),2), "avg_age": round(float(a["avg_age"]),1),
            "players": int(a["players"]),
        },
    })

# ── PLAYERS ───────────────────────────────────────────────────────────────────

@app.route("/api/players")
def players():
    df, err, code = _require_df()
    if err: return err, code
    name   = request.args.get("name","").lower().strip()
    squad  = request.args.get("squad","")
    pos    = request.args.get("pos","")
    sort   = request.args.get("sort","Gls")
    page   = int(request.args.get("page",1))
    per    = 50

    sub = df.copy()
    if name:  sub = sub[sub["Player"].str.lower().str.contains(name, na=False)]
    if squad: sub = sub[sub["Squad"] == squad]
    if pos:   sub = sub[sub["PosSimple"] == pos]
    if sort in FIVE_VARS: sub = sub.sort_values(sort, ascending=False)

    total = len(sub)
    sub = sub.iloc[(page-1)*per : page*per]
    cols = ["Player","Squad","PosSimple","Nation","Age","Min","Gls","Ast","xG"]
    sub = sub[cols].rename(columns={"PosSimple":"Pos"})
    sub = sub.where(pd.notnull(sub), None)

    return jsonify({
        "total": total, "page": page, "per": per,
        "rows": sub.to_dict("records"),
        "positions": sorted(df["PosSimple"].unique().tolist()),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
