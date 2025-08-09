import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import nltk, os

nltk.download('punkt')
df = pd.read_csv("/content/drive/MyDrive/codsoft_task3/RestRecom.csv")
df.fillna("", inplace=True)

def extract_rating(text):
    match = re.search(r'(\d+(?:\.\d+)?)', text)  
    return float(match.group(1)) if match else None

def clean_review_counts(text):
    try:
        text = str(text).replace(",", "").replace("reviews", "").strip()
        return int(text)
    except:
        return None


df['Rating'] = df['Reviews'].apply(extract_rating)
df['ReviewCounts'] = df['No of Reviews'].apply(clean_review_counts)
df['merged'] = df['Name'] + " " + df['Type'] + " " + df['Reviews'] + " " + df['Comments']

model = SentenceTransformer('all-MiniLM-L6-v2')


st.set_page_config("Restaurant Recommendation System", layout="centered")
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    font-family: 'Poppins', sans-serif;
    color: #f5f5f5;
}

.title {
    text-align: center;
    font-size: 2.2rem;
    color: #ff4d6d;
    margin-bottom: 1rem;
}

.result {
    background-color: rgba(255, 255, 255, 0.07);
    padding: 1.2rem;
    border-left: 4px solid #ff4d6d;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(255, 77, 109, 0.2);
}
.result:hover {
    background-color: rgba(255, 255, 255, 0.1);
}
a {
    color: #00d2ff;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'>Smart Restaurant Recommendation System</h1>", unsafe_allow_html=True)


location_options = sorted(df['Location'].dropna().unique())
type_options = sorted(df['Type'].dropna().unique())
location = st.selectbox("Enter your Preferred Location", location_options)
cuisine = st.selectbox("Select your Cuisine Type", type_options)
rating = st.selectbox("Minimum Rating", ["Any", "4+", "3+", "2+"])
review_count = st.selectbox("Number of Reviews for the selected restaurant", ["Any", "100+", "500+", "1000+"])

def generate_query(loc, cui, rat, rc):
    query = f"{cui} restaurants in {loc}"
    if rat != "Any":
        query += f" with rating {rat} stars"
    if rc != "Any":
        query += f" and {rc} reviews"
    return query

def filter_df(rating_filter, review_filter, location, cuisine):
    filtered = df.copy()
    location = location.lower().strip()
    cuisine = cuisine.lower().strip()

    filtered = filtered[filtered['Location'].str.lower().str.contains(location)]
    filtered = filtered[filtered['Type'].str.lower().str.contains(cuisine)]
    if rating_filter != "Any":
        filtered = filtered[filtered['Rating'] >= int(rating_filter[0])]
    if review_filter != "Any":
        min_reviews = int(review_filter.replace("+", "").replace(",", ""))
        filtered = filtered[filtered['ReviewCounts'] >= min_reviews]

    return filtered
if st.button("Get Recommendations"):
    with st.spinner("Finding the best matches based on filters!!"):
        query = generate_query(location, cuisine, rating, review_count)
        query_embedding = model.encode(query, convert_to_tensor=True)
        filtered_df = filter_df(rating, review_count, location, cuisine)

        if filtered_df.empty:
            st.warning("No restaurants matches for the selected filters.")
        else:
            filtered_embeddings = model.encode(filtered_df['merged'].tolist(), convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, filtered_embeddings)[0]
            top_indices = cosine_scores.argsort(descending=True)[:3]

            st.subheader("Recommended restaurants for you..")
            for idx in top_indices:
                row = filtered_df.iloc[int(idx)]
                st.markdown(f"""
                    <div class="result">
                        <h4>{row['Name']}</h4>
                        <p><strong>Type:</strong> {row['Type']}</p>
                        <p><strong>Rating:</strong> {row['Rating']} ⭐</p>
                        <p><strong>Reviews:</strong> {row['ReviewCounts']}</p>
                        <p><strong>Comment:</strong> {row['Comments']}</p>
                        <p><strong>Contact:</strong> {row['Contact Number']}</p>
                        <p>
                            <a href="{row['Menu']}" target="_blank">Click here for Menu</a></p>
                            <a href="{row['Trip_advisor Url']}" target="_blank">TripAdvisor</a>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

