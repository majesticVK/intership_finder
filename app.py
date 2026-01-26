import os
import json
import datetime
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Embeddings + FAISS
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

st.set_page_config(page_title="Internship Recommender (Hybrid)", layout="wide")
st.title("Internship Recommender")

# -------------------------
# Paths
# -------------------------
BASE = Path(__file__).resolve().parent
DATA_CSV = BASE / "internships.csv"

# -------------------------
# Load CSV and fix columns
# -------------------------
intern_df = pd.read_csv(DATA_CSV)

# Map columns to internal standard names
intern_df.rename(columns={
    "Company Name": "company",
    "Role": "title",
    "Description": "desc",
    "Skills Needed": "skills_needed",
    "Stipend": "stipend",
    "Timeframe": "duration_months"
}, inplace=True)

# Fill missing values
intern_df.fillna("", inplace=True)

# Combine text for TF-IDF / embeddings
intern_df["title_description"] = intern_df["title"] + ". " + intern_df["desc"] + ". Skills: " + intern_df["skills_needed"]

# -------------------------
# TF-IDF
# -------------------------
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"https?:\/\/\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return text.strip()

TF_VEC = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
TF_TFIDF = TF_VEC.fit_transform(intern_df["title_description"].map(preprocess_text))

# -------------------------
# Embeddings + FAISS
# -------------------------
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_embed_model():
    return SentenceTransformer(EMB_MODEL_NAME)

@st.cache_resource
def build_embeddings(df, text_col="title_description"):
    model = get_embed_model()
    texts = df[text_col].tolist()
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, embs

if FAISS_AVAILABLE:
    emb_index, embs = build_embeddings(intern_df)
else:
    emb_index, embs = None, None

# -------------------------
# Hybrid search
# -------------------------
def hybrid_search(profile_text: str, top_k=5):
    # TF-IDF
    q = preprocess_text(profile_text)
    q_vec = TF_VEC.transform([q])
    cos_sim = linear_kernel(q_vec, TF_TFIDF).flatten()

    # Embeddings
    model = get_embed_model()
    q_emb = model.encode([profile_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = emb_index.search(q_emb, top_k)
    emb_sims = {i: float(d) for d, i in zip(D[0], I[0])}

    results = []
    for idx, row in intern_df.iterrows():
        tfidf_score = float(cos_sim[idx])
        emb_score = emb_sims.get(idx, 0.0)
        final_score = 0.5 * tfidf_score + 0.5 * emb_score
        results.append((final_score, row.to_dict()))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = []
    for score, row in results[:top_k]:
        row["match_score"] = score
        top_results.append(row)
    return top_results

# -------------------------
# Streamlit UI
# -------------------------
st.subheader("Candidate Profile")
skills = st.text_area("Enter your skills and background", height=120)
uploaded = st.file_uploader("Upload resume (txt)", type=["txt"])
if uploaded:
    try:
        uploaded_text = uploaded.read().decode("utf-8", errors="ignore")
        st.info("Resume loaded (preview):")
        st.code(uploaded_text[:500])
        skills += "\n" + uploaded_text
    except Exception:
        st.warning("Could not read uploaded file.")

year = st.selectbox("Current study year", [1,2,3,4])
pref_location = st.text_input("Preferred location (optional)")
exp_stipend = st.text_input("Expected stipend (optional)")

top_n = st.slider("Top N to show", 1, 10, 5)

if st.button("Recommend"):
    if not skills.strip():
        st.warning("Enter profile info or upload resume")
    else:
        profile_text = f"Skills: {skills}\nYear: {year}\nLocation: {pref_location}\nExpected stipend: {exp_stipend}"
        results = hybrid_search(profile_text, top_k=top_n)

        for r in results:
            st.markdown(f"### {r.get('title')} — {r.get('company')}")
            st.write(
                f"Location: {r.get('location','-')} • "
                f"Stipend: {r.get('stipend','-')} • "
                f"Duration: {r.get('duration_months','-')} mo"
            )
            st.write(f"*Match score:* {r.get('match_score',0):.3f}")
            st.write(r.get('desc',''))
            st.write("*Skills required:*", r.get("skills_needed","-"))
