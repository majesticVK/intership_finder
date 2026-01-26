# app.py
import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# -------------------------
# ML & NLP
# -------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# -------------------------
# Gemini
# -------------------------
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyBXyzEksEWq3IDG9Rh66-uoaTCk1F603d8")
    client = genai  # assign client
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False
    client = None

# -------------------------
# Paths
# -------------------------
BASE = Path(__file__).resolve().parent
INTERNS_CSV = BASE / "internshipshala.csv"
FORMATTED_CSV = BASE / "formatted_jobs.csv"
MODEL_FILE = BASE / "ml_shortlister.pkl"
FEEDBACK_CSV = BASE / "feedback.csv"

# -------------------------
# Load datasets
# -------------------------
internships = pd.read_csv(INTERNS_CSV).rename(columns=str.strip)
formatted_jobs = pd.read_csv(FORMATTED_CSV).rename(columns=str.strip)

# Map some job titles to unified categories
job_mapping = {
    "HR Generalist":"HR Specialist",
    "Field Sales":"Sales Representative",
    "Architecture":"UX Designer",
    "Fashion Design":"Graphic Designer",
    "Operations":"Project Manager",
    "Business Development (Sales)":"Sales Representative",
    "Video Editor":"Graphic Designer",
    "Sales & Communication Manager":"Marketing Manager"
}
internships["mapped_job_title"] = internships["job"].map(job_mapping).fillna(internships["job"])

# Merge with formatted jobs
merge_right = "job_title" if "job_title" in formatted_jobs.columns else formatted_jobs.columns[0]
merged_jobs = internships.merge(formatted_jobs, left_on="mapped_job_title", right_on=merge_right, how="left")
merged_jobs["title_description"] = merged_jobs["job"].fillna("") + ". " + merged_jobs["Skills_required"].fillna("")

# -------------------------
# TF-IDF + Embeddings
# -------------------------
TF_VEC = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
TF_TFIDF = TF_VEC.fit_transform(merged_jobs["title_description"].map(lambda x: str(x).lower()))

if FAISS_AVAILABLE:
    EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(EMB_MODEL_NAME)
    job_embs = embed_model.encode(merged_jobs["title_description"].tolist(), normalize_embeddings=True)

# -------------------------
# Feature calculation
# -------------------------
def calc_features(candidate, job):
    cand_skills = set([s.strip().lower() for s in candidate["skills"].split(",")])
    job_skills = set([s.strip().lower() for s in str(job.get("Skills_required","")).split(",")])
    skills_match = len(cand_skills & job_skills) / max(len(job_skills),1)
    
    exp_match = 1 if candidate["year"] >= job.get("eligibility_year_min",1) else 0
    loc_match = 1 if candidate.get("location","").lower() == job.get("location","").lower() else 0
    salary_match = 1
    if "expected_stipend" in candidate and pd.notna(job.get("salary_avg")):
        salary_match = 1 if candidate.get("expected_stipend",0) <= job["salary_avg"] else 0

    tfidf_score = cosine_similarity(TF_VEC.transform([candidate["skills"]]),
                                    TF_TFIDF[merged_jobs.index.get_loc(job.name)])[0][0] if job.name in merged_jobs.index else 0
    emb_score = 0
    if FAISS_AVAILABLE and job.name in merged_jobs.index:
        emb_score = (embed_model.encode([candidate["skills"]], normalize_embeddings=True) @ job_embs[job.name].T)[0]

    return {
        "skills_match":skills_match,
        "exp_match":exp_match,
        "location_match":loc_match,
        "salary_match":salary_match,
        "tfidf_score":tfidf_score,
        "emb_score":emb_score
    }

# -------------------------
# Train ML model on merged jobs
# -------------------------
if MODEL_FILE.exists():
    model = joblib.load(MODEL_FILE)
else:
    # Create mock candidates for training
    candidates = pd.DataFrame([
        {"candidate_id":1,"name":"Arjun","skills":"Python, SQL, Machine Learning","year":2,"location":"Delhi"},
        {"candidate_id":2,"name":"Meera","skills":"Marketing, Communication, SEO","year":3,"location":"Mumbai"},
        {"candidate_id":3,"name":"Ravi","skills":"UI/UX Design, Figma, Creativity","year":1,"location":"Bangalore"},
    ])
    train_rows = []
    for idx, job in merged_jobs.iterrows():
        for _, cand in candidates.iterrows():
            feat = calc_features(cand, job)
            label = 1 if feat["skills_match"]>=0.5 and feat["exp_match"]==1 else 0
            feat["label"] = label
            train_rows.append(feat)
    train_df = pd.DataFrame(train_rows)
    X_train = train_df[["skills_match","exp_match","location_match","salary_match","tfidf_score","emb_score"]]
    y_train = train_df["label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Internship Shortlister", layout="wide")
st.title("AI + Rule-based Internship Recommender")

with st.sidebar:
    st.header("Candidate Info")
    skills = st.text_area("Skills & Background", height=120)
    year = st.selectbox("Current Year", [1,2,3,4])
    location = st.text_input("Preferred Location")
    expected_stipend = st.number_input("Expected Stipend (optional)", min_value=0, value=0)

    uploaded = st.file_uploader("Upload Resume (.txt)", type=["txt"])
    if uploaded:
        try:
            text = uploaded.read().decode("utf-8")
            skills += "\n" + text
        except:
            st.warning("Could not read uploaded file.")

candidate_profile = {"skills":skills,"year":year,"location":location,"expected_stipend":expected_stipend}

# -------------------------
# Recommend Internships
# -------------------------
if st.button("Recommend"):
    scored_jobs = []
    for idx, job in internships.iterrows():  # use all internships, not just merged
        feat = calc_features(candidate_profile, job)
        X_feat = pd.DataFrame([feat])
        score = 0
        if job["mapped_job_title"] in merged_jobs["mapped_job_title"].values:
            # ML + semantic score
            score = model.predict_proba(X_feat)[0][1]
            score = 0.6*score + 0.4*(feat["tfidf_score"] + feat["emb_score"])/2
        else:
            # Only semantic similarity
            score = (feat["tfidf_score"] + feat["emb_score"])/2 if FAISS_AVAILABLE else feat["tfidf_score"]
        feat["match_score"] = score
        scored_jobs.append((score, job, feat))

    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    top_n = 10
    for rank, (score, job, feat) in enumerate(scored_jobs[:top_n],1):
        st.markdown(f"### {job['job']} — {job.get('company_name','-')}")
        st.write(f"Location: {job.get('location','-')} • Stipend: {job.get('salary','-')}")
        st.write(f"*Match score:* {score:.3f}")
        st.write(f"Required Skills: {job.get('Skills_required','-')}")
        st.write(f"Candidate features: {feat}")

        # Gemini explanation
        if GEMINI_AVAILABLE:
            prompt = f"""
You are an internship recommender.
Candidate skills: {candidate_profile['skills']}
Job: {job['job']} at {job.get('company_name','-')}
Skills required: {job.get('Skills_required','-')}
Explain in 2 sentences why this is a good match.
"""
            try:
                resp = client.models.generate_content(
                    model="models/gemini-1.5-flash",
                    contents=prompt,
                    config={"temperature":0.3, "max_output_tokens":200}
                )
                explanation = getattr(resp, "text", str(resp))
            except:
                explanation = "Gemini unavailable"
        else:
            explanation = "Rule + ML + semantic match."

        st.info(explanation)

         

# -------------------------
# Feedback functions
# -------------------------
def save_feedback(candidate_profile, job, vote):
    df_fb = pd.DataFrame([{
        "profile": candidate_profile["skills"][:500],
        "job": job['job'],
        "vote": vote
    }])
    df_fb.to_csv(FEEDBACK_CSV, mode="a", index=False, header=not FEEDBACK_CSV.exists())
    st.success("Saved feedback")
    retrain_model()

def retrain_model():
    if not FEEDBACK_CSV.exists():
        return
    fb = pd.read_csv(FEEDBACK_CSV)
    train_rows = []
    for idx, job in merged_jobs.iterrows():
        job_skills = set([s.strip().lower() for s in str(job.get("Skills_required","")).split(",")])
        for _, row in fb.iterrows():
            cand_skills = set([s.strip().lower() for s in row["profile"].split(",")])
            skills_match = len(cand_skills & job_skills) / max(len(job_skills),1)
            feat = {
                "skills_match": skills_match,
                "exp_match": 1,
                "location_match": 0,
                "salary_match": 1,
                "tfidf_score": cosine_similarity(
                    TF_VEC.transform([row["profile"]]),
                    TF_TFIDF[merged_jobs.index.get_loc(job.name)]
                )[0][0],
                "emb_score": 0,
                "label": row["vote"]
            }
            train_rows.append(feat)
    if train_rows:
        train_df = pd.DataFrame(train_rows)
        X_train = train_df[["skills_match","exp_match","location_match","salary_match","tfidf_score","emb_score"]]
        y_train = train_df["label"]
        global model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
        st.success("ML model retrained based on feedback")

        # Feedback buttons
        if st.button(f"👍 Yes_{job['job']}", key=f"yes_{job['job']}"):
            save_feedback(candidate_profile, job, vote=1)
        if st.button(f"👎 No_{job['job']}", key=f"no_{job['job']}"):
            save_feedback(candidate_profile, job, vote=0)

    