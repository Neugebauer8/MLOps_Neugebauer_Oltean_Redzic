# Phase 1 — Classic ML + GitHub Pages Deployment

## What this phase delivers

A browser-only predictor that estimates a professional footballer's market value (in €) from 36 performance statistics. No backend. No server. The model is trained offline with scikit-learn, exported as `model.json`, and the entire prediction loop runs in the user's browser.

| File | Role |
|---|---|
| `phase1.ipynb` | Fully documented training notebook (load → EDA → train → export) |
| `player_stats.csv` | FIFA 21 dataset — 5,682 rows, 40 features + target |
| `model.json` | Exported linear-regression weights + preprocessing params + metrics |
| `index.html` | Single-page web app |
| `style.css` | Editorial / sports-data styling |
| `predict.js` | Slider generation + client-side inference |
| `README.md` | This document |

---

## Dataset

- **Source**: [FIFA 21 Player Stats Database on Kaggle](https://www.kaggle.com/datasets/bryanb/fifa-player-stats-database)
- **Rows**: 5,682 professional players
- **Target**: `value` (parsed from FIFA currency strings into plain €)
- **Features used**: 36 numeric attributes (physical, technical, mental, goalkeeping)
- **Dropped**: `player`, `country`, `club` (identifiers), `marking` (100 % missing)

---

## Modelling choices

| Decision | Rationale |
|---|---|
| `log1p(value)` as target | Raw values span €4K – €153.5M; a multiplicative model fits the market much better than an additive one. |
| `StandardScaler` | Keeps coefficients interpretable and comparable across features. |
| Plain `LinearRegression` | Phase 1 requires a JSON-exportable model that runs without re-training in the browser. Coefficients + intercept + scaler stats = everything the UI needs. |
| 80 / 20 train-test split, seed 42 | Reproducible metrics. |

Test-set performance:

| Metric | Value |
|---|---|
| **MAE** | ≈ €1.12 M |
| **RMSE** | ≈ €4.75 M |
| **R² (log-space)** | ≈ 0.79 |
| **R² (€ scale)** | ≈ 0.61 |

The log-space R² is the honest number — the € R² is dragged down by the handful of €100M+ superstars, whose squared errors dominate. Phase 2 introduces stronger models (Ridge, Random Forest, etc.) that we compare rigorously via MLflow.

---

## How the browser inference works

`predict.js` implements the exact same pipeline sklearn used, but in ~20 lines of JavaScript:

```js
// 1. Assemble raw feature vector in the training order
const raw = model.feature_names.map(f => currentValues[f]);

// 2. Standard-scale (reproduces sklearn's StandardScaler)
const scaled = raw.map((x, i) => (x - model.feature_means[i]) / model.feature_stds[i]);

// 3. Linear combination in log-space
let yLog = model.intercept;
for (let i = 0; i < scaled.length; i++) yLog += scaled[i] * model.coef[i];

// 4. Invert the log1p transform
const yEur = Math.expm1(yLog);
```

The ML.js library is loaded from a CDN (per the project spec), but because our model is a classic linear regression, we perform the dot-product directly — no re-training inside the browser is required.

---

## Running locally

```bash
# No build step is needed. Any static HTTP server works.
cd phase1
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

Opening `index.html` directly via `file://` works too, but some browsers block `fetch('model.json')` from the local filesystem, so a tiny HTTP server is safer.

---

## Deployment via GitHub Pages

```bash
# From the phase1/ folder
git init
git add index.html style.css predict.js model.json README.md
git commit -m "Phase 1 — FIFA player value predictor"

# Create a new public repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/<your-username>/mlops-fifa-player-value.git
git push -u origin main
```

Then in the GitHub UI:

1. **Settings → Pages**
2. **Source:** `Deploy from a branch`
3. **Branch:** `main`, **Folder:** `/ (root)`
4. **Save**

After ~60 seconds the site is live at `https://<your-username>.github.io/mlops-fifa-player-value/`. Paste that URL at the top of this README.

---

## Using the app

1. Pick a tab (Physical / Mental / Attacking / Skill / Movement / Defending / Goalkeeper) and tune the sliders for whichever stats matter for the player you have in mind.
2. All sliders the user does **not** touch remain at the training-set median — a safe default. The model still uses all 36 features every time; the tabs just make the UI manageable.
3. Click one of the **preset chips** (Superstar striker, Elite defender, Top goalkeeper, etc.) to auto-fill a realistic profile and see a plausible market value instantly.
4. Leave **Live update** on to feel the market react as you drag, or turn it off and use the **Predict** button.

---

## Data flow

```
player_stats.csv
      │
      ▼
  phase1.ipynb   ───►  sklearn.LinearRegression  ───►  model.json
                                                           │
                              ┌────────────────────────────┘
                              ▼
                         index.html ◄── style.css
                              │
                              ▼
                         predict.js (vanilla-JS dot-product)
                              │
                              ▼
                      €-valued prediction
```


This project was created by the students Moritz Neugebauer, Nico Oltean, Eldin Redzic