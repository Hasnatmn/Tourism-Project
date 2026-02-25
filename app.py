import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Tourism Recommendation System", layout="wide")

st.title("Tourism Experience Recommendation System")
st.write("User-Based Collaborative Filtering using Cosine Similarity")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_df.csv")
    return df

df = load_data()

# -------------------------------
# CREATE USER-ITEM MATRIX
# -------------------------------
user_item_matrix = df.pivot_table(
    index='UserId',
    columns='AttractionId',
    values='Rating'
)

user_item_filled = user_item_matrix.fillna(0)

# -------------------------------
# COSINE SIMILARITY
# -------------------------------
user_similarity = cosine_similarity(user_item_filled)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_filled.index,
    columns=user_item_filled.index
)

# -------------------------------
# FUNCTIONS
# -------------------------------
def get_similar_users(user_id, top_n=5):

    if user_id not in user_similarity_df.index:
        return None

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)

    return similar_users.head(top_n)


def recommend_attractions(user_id, top_n_users=5, top_n_recommendations=5):

    similar_users = get_similar_users(user_id, top_n_users)

    if similar_users is None:
        return None

    # Attractions already rated by target user
    user_rated = user_item_matrix.loc[user_id].dropna().index

    # Get ratings from similar users
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    # Average rating
    recommendation_scores = similar_users_ratings.mean().sort_values(ascending=False)

    # Remove already rated attractions
    recommendation_scores = recommendation_scores.drop(user_rated, errors='ignore')

    return recommendation_scores.head(top_n_recommendations).index


def get_recommendation_details(user_id, top_n_users=5, top_n_recommendations=5):

    recommended_ids = recommend_attractions(user_id, top_n_users, top_n_recommendations)

    if recommended_ids is None:
        return None

    details = df[df['AttractionId'].isin(recommended_ids)][[
        'AttractionId',
        'item_Attraction',
        'item_AttractionAddress',
        'city_CityName',
        'country_CountryName'
    ]].drop_duplicates().reset_index(drop=True)

    return details


# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Recommendation Settings")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=int(df['UserId'].min()),
    max_value=int(df['UserId'].max()),
    step=1
)

top_n_users = st.sidebar.slider("Number of Similar Users", 1, 10, 5)
top_n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# -------------------------------
# BUTTON
# -------------------------------
if st.button("Get Recommendations"):

    recommendations = get_recommendation_details(
        user_id,
        top_n_users,
        top_n_recommendations
    )

    if recommendations is None or recommendations.empty:
        st.error("No recommendations found for this user.")
    else:
        st.success("Recommended Attractions")
        st.dataframe(recommendations, use_container_width=True)

        st.subheader("Similar Users")
        similar_users = get_similar_users(user_id, top_n_users)
        st.write(similar_users)