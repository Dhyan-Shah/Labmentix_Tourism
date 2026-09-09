"""
Tourism Experience Analytics — Streamlit App
Project 9 | Labmentix DS & AI/ML Internship

Run with: streamlit run app.py
Expects to sit in the same folder as:
  - data/ (Transaction.xlsx, User.xlsx, City.xlsx, Country.xlsx, Region.xlsx,
            Continent.xlsx, Type.xlsx, Item.xlsx, Mode.xlsx)
  - best_regression_model.pkl, best_classification_model.pkl, label_encoders.pkl,
    scaler.pkl, target_label_encoder.pkl, user_item_matrix.pkl, content_similarity.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🧭", layout="wide")

DATA_DIR = "data/"
CATEGORICAL_COLS = ['Continent', 'Country', 'Region', 'AttractionType']
NUMERIC_FEATURE_COLS = ['VisitYear', 'VisitMonth', 'user_avg_rating', 'user_visit_count', 'attraction_visit_count']
FEATURE_COLS = [c + '_enc' for c in CATEGORICAL_COLS] + [c + '_scaled' for c in NUMERIC_FEATURE_COLS]


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource
def load_models():
    """Load the artifacts saved at the end of Section 7 in the notebook."""
    artifacts = {}
    try:
        artifacts['reg_model'] = joblib.load('best_regression_model.pkl')
        artifacts['clf_model'] = joblib.load('best_classification_model.pkl')
        artifacts['label_encoders'] = joblib.load('label_encoders.pkl')
        artifacts['scaler'] = joblib.load('scaler.pkl')
        artifacts['target_le'] = joblib.load('target_label_encoder.pkl')
        artifacts['user_item_matrix'] = joblib.load('user_item_matrix.pkl')
        artifacts['content_similarity'] = joblib.load('content_similarity.pkl')
        return artifacts, None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_data
def load_raw_data():
    """Load the raw lookup tables for populating dropdowns and displaying attraction names."""
    city_df = pd.read_excel(DATA_DIR + 'City.xlsx')
    country_df = pd.read_excel(DATA_DIR + 'Country.xlsx')
    region_df = pd.read_excel(DATA_DIR + 'Region.xlsx')
    continent_df = pd.read_excel(DATA_DIR + 'Continent.xlsx')
    type_df = pd.read_excel(DATA_DIR + 'Type.xlsx')
    item_df = pd.read_excel(DATA_DIR + 'Item.xlsx')
    transaction_df = pd.read_excel(DATA_DIR + 'Transaction.xlsx')

    for df, col in [(city_df, 'CityName'), (country_df, 'Country'), (region_df, 'Region'),
                     (continent_df, 'Continent'), (type_df, 'AttractionType')]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return {
        'city': city_df, 'country': country_df, 'region': region_df,
        'continent': continent_df, 'type': type_df, 'item': item_df,
        'transaction': transaction_df
    }


@st.cache_data
def build_attraction_lookup(raw):
    """Attraction-level table with type, name, and historical popularity/rating stats."""
    item_type = raw['item'].merge(raw['type'], on='AttractionTypeId', how='left')
    agg = raw['transaction'].groupby('AttractionId').agg(
        attraction_avg_rating=('Rating', 'mean'),
        attraction_visit_count=('TransactionId', 'count')
    ).reset_index()
    lookup = item_type.merge(agg, on='AttractionId', how='left')
    name_col = 'Attraction' if 'Attraction' in lookup.columns else lookup.columns[1]
    lookup = lookup.rename(columns={name_col: 'AttractionName'}) if name_col != 'AttractionName' else lookup
    return lookup


@st.cache_data
def compute_global_stats(raw):
    """Fallback averages for numeric features when we don't have per-user/per-attraction history
    (e.g. a brand-new user with no prior visits)."""
    user_stats = raw['transaction'].groupby('UserId').agg(
        user_avg_rating=('Rating', 'mean'),
        user_visit_count=('TransactionId', 'count')
    )
    attraction_stats = raw['transaction'].groupby('AttractionId').agg(
        attraction_visit_count=('TransactionId', 'count')
    )
    return {
        'user_avg_rating': user_stats['user_avg_rating'].mean(),
        'user_visit_count': user_stats['user_visit_count'].mean(),
        'attraction_visit_count': attraction_stats['attraction_visit_count'].mean(),
        'user_stats': user_stats,
        'attraction_stats': attraction_stats
    }


@st.cache_resource
def build_knn_model(_user_item_matrix):
    """Rebuilt on app startup rather than pickled — cheap to fit and keeps the app
    self-contained without an extra large artifact file."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    sparse_matrix = csr_matrix(_user_item_matrix.fillna(0).values)
    user_ids = _user_item_matrix.index.tolist()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    model.fit(sparse_matrix)
    return model, sparse_matrix, user_ids, user_id_to_idx


# ============================================================
# ENCODING HELPERS
# ============================================================

def safe_label_encode(le, value):
    """Transform a single value, falling back to the encoder's most common class
    if the value wasn't seen during training (avoids crashing on unseen categories)."""
    try:
        return le.transform([value])[0]
    except ValueError:
        return le.transform([le.classes_[0]])[0]


def build_feature_vector(continent, country, region, attraction_type,
                          visit_year, visit_month,
                          user_avg_rating, user_visit_count, attraction_visit_count,
                          label_encoders, scaler):
    cat_values = {'Continent': continent, 'Country': country, 'Region': region, 'AttractionType': attraction_type}
    encoded = {}
    for col in CATEGORICAL_COLS:
        le = label_encoders[col]
        encoded[col + '_enc'] = safe_label_encode(le, cat_values[col])

    numeric_row = pd.DataFrame([[visit_year, visit_month, user_avg_rating, user_visit_count, attraction_visit_count]],
                                columns=NUMERIC_FEATURE_COLS)
    scaled = scaler.transform(numeric_row)[0]
    for i, col in enumerate(NUMERIC_FEATURE_COLS):
        encoded[col + '_scaled'] = scaled[i]

    return pd.DataFrame([encoded])[FEATURE_COLS]


# ============================================================
# RECOMMENDATION FUNCTIONS (mirrors notebook Section 7.3)
# ============================================================

def collaborative_recommend(user_id, user_item_matrix, knn_model, sparse_matrix, user_ids, user_id_to_idx,
                             n=5, n_similar_users=10):
    if user_id not in user_id_to_idx:
        return pd.Series(dtype=float)
    idx = user_id_to_idx[user_id]

    distances, indices = knn_model.kneighbors(sparse_matrix[idx], n_neighbors=n_similar_users + 1)
    similarities = 1 - distances.flatten()
    neighbor_idx = indices.flatten()

    keep = neighbor_idx != idx
    neighbor_idx = neighbor_idx[keep][:n_similar_users]
    similarities = similarities[keep][:n_similar_users]

    if len(neighbor_idx) == 0 or similarities.sum() == 0:
        return pd.Series(dtype=float)

    similar_user_ids = [user_ids[i] for i in neighbor_idx]
    similar_users_ratings = user_item_matrix.loc[similar_user_ids]

    weighted_ratings = similar_users_ratings.T.dot(similarities) / similarities.sum()
    already_rated = user_item_matrix.loc[user_id].dropna().index
    weighted_ratings = weighted_ratings.drop(labels=already_rated, errors='ignore')
    return weighted_ratings.sort_values(ascending=False).head(n)


def content_based_recommend(attraction_id, content_similarity_df, n=5):
    if attraction_id not in content_similarity_df.index:
        return pd.Series(dtype=float)
    scores = content_similarity_df[attraction_id].sort_values(ascending=False)
    scores = scores.drop(labels=[attraction_id], errors='ignore')
    return scores.head(n)


def hybrid_recommend(user_id, user_item_matrix, content_similarity_df, knn_model, sparse_matrix,
                      user_ids, user_id_to_idx, n=5, alpha=0.6):
    collab = collaborative_recommend(user_id, user_item_matrix, knn_model, sparse_matrix,
                                      user_ids, user_id_to_idx, n=20)
    if collab.empty:
        return pd.Series(dtype=float)

    user_ratings = user_item_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return collab.head(n)

    top_rated_attraction = user_ratings.idxmax()
    content = content_based_recommend(top_rated_attraction, content_similarity_df, n=20)

    combined = pd.DataFrame({'collab': collab, 'content': content}).fillna(0)
    combined['score'] = alpha * combined['collab'] + (1 - alpha) * combined['content']
    return combined['score'].sort_values(ascending=False).head(n)


# ============================================================
# APP LAYOUT
# ============================================================

st.title("🧭 Tourism Experience Analytics")
st.caption("Regression • Classification • Recommendation — Project 9, Labmentix DS & AI/ML Internship")

artifacts, load_error = load_models()

if load_error:
    st.error(
        f"Couldn't load saved model artifacts ({load_error}). "
        "Make sure this app sits in the same folder where the notebook saved the .pkl files, "
        "and that you've run Section 7.6 (Save best models for deployment) at least once."
    )
    st.stop()

raw = load_raw_data()
attraction_lookup = build_attraction_lookup(raw)
global_stats = compute_global_stats(raw)
knn_model, sparse_matrix, user_ids, user_id_to_idx = build_knn_model(artifacts['user_item_matrix'])

tab_overview, tab_mode, tab_rating, tab_recs = st.tabs(
    ["📊 Overview", "🎯 Predict Visit Mode", "⭐ Predict Rating", "🗺️ Get Recommendations"]
)

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(raw['transaction']):,}")
    c2.metric("Users", f"{raw['transaction']['UserId'].nunique():,}")
    c3.metric("Attractions", f"{raw['item']['AttractionId'].nunique():,}")
    c4.metric("Avg. Rating", f"{raw['transaction']['Rating'].mean():.2f} / 5")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Rating distribution**")
        st.bar_chart(raw['transaction']['Rating'].value_counts().sort_index())
    with col_b:
        st.markdown("**Top attraction types by visit count**")
        top_types = attraction_lookup.groupby('AttractionType')['attraction_visit_count'].sum() \
            .sort_values(ascending=False).head(10)
        st.bar_chart(top_types)

# ------------------------------------------------------------
# TAB 2: PREDICT VISIT MODE (classification)
# ------------------------------------------------------------
with tab_mode:
    st.subheader("Predict how someone is likely visiting an attraction")
    st.caption("Business • Family • Couples • Friends • Solo — based on where they're from and what they're visiting.")

    le = artifacts['label_encoders']
    continent_options = sorted(le['Continent'].classes_)
    col1, col2 = st.columns(2)

    with col1:
        continent = st.selectbox("Continent", continent_options, key="mode_continent")
        country_options = sorted(raw['country'][raw['country']['ContinentId'].isin(
            raw['continent'][raw['continent']['Continent'] == continent]['ContinentId']
        )]['Country'].unique()) if 'ContinentId' in raw['country'].columns else sorted(le['Country'].classes_)
        country = st.selectbox("Country", country_options if len(country_options) else sorted(le['Country'].classes_),
                                key="mode_country")
        region_options = sorted(le['Region'].classes_)
        region = st.selectbox("Region", region_options, key="mode_region")

    with col2:
        attraction_type = st.selectbox("Attraction Type", sorted(le['AttractionType'].classes_), key="mode_type")
        visit_year = st.number_input("Visit Year", min_value=2010, max_value=2030,
                                      value=int(raw['transaction']['VisitYear'].max()), key="mode_year")
        visit_month = st.slider("Visit Month", 1, 12, 6, key="mode_month")

    if st.button("Predict Visit Mode", type="primary"):
        X = build_feature_vector(
            continent, country, region, attraction_type, visit_year, visit_month,
            global_stats['user_avg_rating'], global_stats['user_visit_count'], global_stats['attraction_visit_count'],
            artifacts['label_encoders'], artifacts['scaler']
        )

        clf_model = artifacts['clf_model']
        target_le = artifacts['target_le']

        if hasattr(clf_model, 'classes_') and set(clf_model.classes_) <= set(range(len(target_le.classes_))):
            pred_encoded = clf_model.predict(X)[0]
            pred_label = target_le.inverse_transform([pred_encoded])[0]
        else:
            pred_label = clf_model.predict(X)[0]

        st.success(f"**Predicted Visit Mode: {pred_label}**")

        if hasattr(clf_model, 'predict_proba'):
            proba = clf_model.predict_proba(X)[0]
            classes = target_le.classes_ if len(proba) == len(target_le.classes_) else clf_model.classes_
            proba_df = pd.DataFrame({'Visit Mode': classes, 'Probability': proba}).sort_values(
                'Probability', ascending=False)
            st.markdown("**Confidence breakdown**")
            st.bar_chart(proba_df.set_index('Visit Mode'))

# ------------------------------------------------------------
# TAB 3: PREDICT RATING (regression)
# ------------------------------------------------------------
with tab_rating:
    st.subheader("Predict the rating a visit is likely to get")

    le = artifacts['label_encoders']
    col1, col2 = st.columns(2)

    with col1:
        r_continent = st.selectbox("Continent", sorted(le['Continent'].classes_), key="rate_continent")
        r_country = st.selectbox("Country", sorted(le['Country'].classes_), key="rate_country")
        r_region = st.selectbox("Region", sorted(le['Region'].classes_), key="rate_region")

    with col2:
        attraction_name_options = attraction_lookup.get(
            'AttractionName', attraction_lookup.iloc[:, 1]
        ).dropna().unique().tolist()
        selected_attraction = st.selectbox("Attraction (optional — uses its history if picked)",
                                            ["(none — use type only)"] + sorted(attraction_name_options),
                                            key="rate_attraction")
        r_type = st.selectbox("Attraction Type", sorted(le['AttractionType'].classes_), key="rate_type")
        r_year = st.number_input("Visit Year", min_value=2010, max_value=2030,
                                  value=int(raw['transaction']['VisitYear'].max()), key="rate_year")
        r_month = st.slider("Visit Month", 1, 12, 6, key="rate_month")

    if st.button("Predict Rating", type="primary"):
        attraction_visit_count = global_stats['attraction_visit_count']
        if selected_attraction != "(none — use type only)":
            name_col = 'AttractionName' if 'AttractionName' in attraction_lookup.columns else attraction_lookup.columns[1]
            match = attraction_lookup[attraction_lookup[name_col] == selected_attraction]
            if not match.empty and pd.notna(match.iloc[0].get('attraction_visit_count')):
                attraction_visit_count = match.iloc[0]['attraction_visit_count']

        X = build_feature_vector(
            r_continent, r_country, r_region, r_type, r_year, r_month,
            global_stats['user_avg_rating'], global_stats['user_visit_count'], attraction_visit_count,
            artifacts['label_encoders'], artifacts['scaler']
        )

        pred_rating = artifacts['reg_model'].predict(X)[0]
        pred_rating = float(np.clip(pred_rating, 1, 5))
        st.success(f"**Predicted Rating: {pred_rating:.2f} / 5**")
        st.progress(pred_rating / 5)

# ------------------------------------------------------------
# TAB 4: RECOMMENDATIONS
# ------------------------------------------------------------
with tab_recs:
    st.subheader("Get personalized attraction recommendations")

    mode = st.radio("I am...", ["An existing user (have a User ID)", "A new visitor (no history yet)"],
                     horizontal=True)

    name_col = 'AttractionName' if 'AttractionName' in attraction_lookup.columns else attraction_lookup.columns[1]

    if mode == "An existing user (have a User ID)":
        selected_user = st.selectbox("Select User ID", user_ids[:2000],
                                      help="Showing a sample of user IDs from the dataset")
        n_recs = st.slider("Number of recommendations", 3, 15, 5)

        if st.button("Get Recommendations", type="primary"):
            recs = hybrid_recommend(
                selected_user, artifacts['user_item_matrix'], artifacts['content_similarity'],
                knn_model, sparse_matrix, user_ids, user_id_to_idx, n=n_recs
            )
            if recs.empty:
                st.warning("Not enough history for this user to generate collaborative recommendations.")
            else:
                rec_df = attraction_lookup[attraction_lookup['AttractionId'].isin(recs.index)][
                    ['AttractionId', name_col, 'AttractionType']].drop_duplicates('AttractionId')
                rec_df = rec_df.set_index('AttractionId').loc[recs.index].reset_index()
                rec_df['Match Score'] = recs.values
                st.dataframe(rec_df[[name_col, 'AttractionType', 'Match Score']], use_container_width=True)

    else:
        preferred_type = st.selectbox("What kind of attraction do you enjoy?",
                                       sorted(attraction_lookup['AttractionType'].dropna().unique()))
        n_recs = st.slider("Number of recommendations", 3, 15, 5, key="new_user_n")

        if st.button("Get Recommendations", type="primary", key="new_user_btn"):
            top_in_type = attraction_lookup[attraction_lookup['AttractionType'] == preferred_type].sort_values(
                'attraction_avg_rating', ascending=False
            ).head(n_recs)
            st.info("Cold-start fallback: showing the highest-rated attractions of your preferred type, "
                    "since collaborative filtering needs some visit history first.")
            st.dataframe(top_in_type[[name_col, 'AttractionType', 'attraction_avg_rating', 'attraction_visit_count']],
                         use_container_width=True)