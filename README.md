# 🧭 Tourism Experience Analytics — Classification, Prediction & Recommendation System

**Project 9 | Data Science & AI/ML Internship — Labmentix**

An end-to-end machine learning system that predicts how a tourist will rate an attraction, classifies
their likely visit mode, and recommends attractions through a hybrid recommendation engine — all
served through a single interactive Streamlit app.

---

## 📌 Problem Statement

Tourism platforms typically show the same generic "top attractions" list to every visitor. This
project asks a more useful question: given who a visitor is and where they're going, can we predict
*how* they're likely visiting (business, family, couples, friends, solo) and *what* they're likely to
enjoy — and use both to personalize recommendations?

The project combines three connected ML tasks on one relational dataset:

| Task | Type | Goal |
|---|---|---|
| Rating Prediction | Regression | Predict a user's rating (1–5) for an attraction |
| Visit Mode Prediction | Classification | Predict how the user is visiting (Business/Family/Couples/Friends/Solo) |
| Attraction Recommendation | Recommender System | Suggest attractions via collaborative + content-based filtering |

---

## 🗂️ Dataset

A 9-table relational (star-schema) dataset, joined via ID keys:

- `Transaction.xlsx` — the fact table: UserId, AttractionId, VisitYear, VisitMonth, VisitMode, Rating
- `User.xlsx` — user demographic/location keys
- `City.xlsx`, `Region.xlsx`, `Country.xlsx`, `Continent.xlsx` — the geographic hierarchy
- `Type.xlsx` — attraction type lookup
- `Item.xlsx` — attraction (item) master data
- `Mode.xlsx` — visit mode ID-to-label lookup

**Data quirks handled during wrangling** (worth noting for anyone reproducing this):
- Source files are `.xlsx`, not `.csv`
- The visit-mode lookup table is named `Mode.xlsx`, not `VisitMode.xlsx`
- `Transaction.xlsx`'s `VisitMode` column holds an **ID** (int), while `Mode.xlsx`'s `VisitMode`
  column holds the **text label** (str) — same column name, different meaning, which breaks a naive
  merge. Resolved by explicitly separating the ID and label columns before joining.
- An `Updated_Item.xlsx` file also exists alongside `Item.xlsx`; this project uses `Item.xlsx` as the
  attraction source of truth.

---

## 🔍 Approach

1. **Know Your Data** — load and inspect all 9 tables individually before joining anything
2. **Understanding Variables** — cardinality checks, categorical inventories, target distribution
3. **Data Wrangling** — standardize categorical text, merge into one master dataframe, handle nulls/duplicates/outliers, build a proper `VisitDate` field
4. **Data Visualization** — rating distribution, visit-mode breakdown, geographic patterns, attraction-type popularity, correlation heatmap
5. **Hypothesis Testing** — ANOVA (rating ~ attraction type, rating ~ visit mode) and chi-square (visit mode ~ continent) to confirm patterns are statistically real, not sampling noise
6. **Feature Engineering** — user- and attraction-level aggregate features, label encoding, scaling, train/test splits — with an explicit check for target leakage in aggregate features (e.g. an attraction's historical average rating computed inclusive of the row being predicted)
7. **Model Implementation**:
   - *Regression*: Linear Regression, Random Forest, XGBoost — compared on RMSE / MAE / R²
   - *Classification*: Random Forest, XGBoost, LightGBM (class-balanced) — compared on Accuracy / Precision / Recall / F1
   - *Recommendation*: collaborative filtering via sparse nearest-neighbors on the user-item matrix, content-based filtering via attraction-type similarity, and a hybrid blend of both, evaluated with Precision@K
8. **Deployment** — a Streamlit app exposing all three models for real-time predictions and recommendations

**Engineering note:** the initial collaborative filtering implementation computed a full dense
user-user similarity matrix, which requires ~8+ GB of memory at this dataset's scale (30,000+ users).
This was replaced with a sparse-matrix `NearestNeighbors` approach that computes similarity on demand
per query, keeping memory proportional to a single row rather than the full user base.

---

## 📊 Results

*(Fill in your final run's numbers here before submission — update from your executed notebook's Section 7.5 output.)*

**Regression (Rating Prediction)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

**Classification (Visit Mode Prediction)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |

**Recommendation Engine**

- Precision@5 (collaborative filtering): —

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, numpy, openpyxl (for `.xlsx` support)
- **Visualization:** matplotlib, seaborn
- **Statistics:** scipy
- **Modeling:** scikit-learn, XGBoost, LightGBM
- **Recommendation:** scikit-learn (`NearestNeighbors`, cosine similarity)
- **Deployment:** Streamlit
- **Model persistence:** joblib

---

## 📁 Project Structure

```
p9/
├── data/
│   ├── Transaction.xlsx
│   ├── User.xlsx
│   ├── City.xlsx
│   ├── Region.xlsx
│   ├── Country.xlsx
│   ├── Continent.xlsx
│   ├── Type.xlsx
│   ├── Item.xlsx
│   └── Mode.xlsx
├── Tourism.ipynb                     # Full notebook (Sections 1–8)
├── app.py                            # Streamlit app
├── best_regression_model.pkl
├── best_classification_model.pkl
├── label_encoders.pkl
├── scaler.pkl
├── target_label_encoder.pkl
├── user_item_matrix.pkl
├── content_similarity.pkl
└── README.md
```

---

## ⚙️ Setup & Installation

```powershell
# 1. Create project folder and venv
mkdir C:\dev\p9
cd C:\dev\p9
python -m venv p9

# 2. Activate the venv
.\p9\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter ipykernel streamlit xgboost lightgbm openpyxl joblib

# 4. Register the Jupyter kernel (for running the notebook in VS Code)
python -m ipykernel install --user --name=p9 --display-name "Python (p9)"
```

Place the 9 `.xlsx` files in a `data/` subfolder, then run `Tourism.ipynb` end-to-end
(**Run All** is recommended over running cells individually, to avoid stale-state errors)
to generate the model artifacts.

---

## ▶️ Running the App

```powershell
streamlit run app.py
```

The app has four tabs:

- **📊 Overview** — dataset metrics and summary charts
- **🎯 Predict Visit Mode** — classify how a visitor is likely traveling
- **⭐ Predict Rating** — estimate the rating a visit is likely to receive
- **🗺️ Get Recommendations** — hybrid recommendations for an existing user, or a cold-start fallback for a new visitor

---

## 🔮 Future Work

- Log experiments (params + metrics) with MLflow across model runs
- Upgrade collaborative filtering from k-NN to matrix factorization (SVD/ALS) for better scale
- Add a proper cold-start onboarding flow (declared preferences at signup) for brand-new users
- Re-verify the leave-one-out fix for `attraction_avg_rating` before reporting final regression metrics
- Consider SMOTE in addition to class weighting if visit-mode classes are heavily imbalanced

---

## 👤 Author

**Dhyan** — Data Science & AI/ML Intern, Labmentix
Project 8 of an ongoing sequential portfolio of end-to-end ML projects.