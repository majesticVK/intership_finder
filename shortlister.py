# shortlister.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import joblib
from sentence_transformers import SentenceTransformer

# ---------- Load datasets ----------
internships = pd.read_csv("internshipshala.csv")
formatted_jobs = pd.read_csv("formatted_jobs.csv")
candidates = pd.read_csv("candidates.csv")

# Map job titles if needed
job_mapping = {"HR Generalist":"HR Specialist","Field Sales":"Sales Representative"}
internships["mapped_job_title"] = internships["job"].map(job_mapping).fillna(internships["job"])

# Merge to enrich info
merged_jobs = internships.merge(formatted_jobs, left_on="mapped_job_title",
                               right_on="job_title", how="left")
merged_jobs["title_description"] = merged_jobs["job"].fillna("") + ". " + merged_jobs["Skills_required"].fillna("") + ". " + merged_jobs["desc"].fillna("")

# ---------- TF-IDF ----------
TF_VEC = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
TF_TFIDF = TF_VEC.fit_transform(merged_jobs["title_description"].map(lambda x: str(x).lower()))

# Optional embeddings
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
job_embs = embed_model.encode(merged_jobs["title_description"].tolist(), normalize_embeddings=True)

# ---------- Feature Calculation ----------
def calc_features(candidate, job):
    cand_skills = set([s.strip().lower() for s in candidate["skills"].split(",")])
    job_skills = set([s.strip().lower() for s in str(job.get("Skills_required","")).split(",")])
    skills_match = len(cand_skills & job_skills) / max(len(job_skills),1)

    exp_match = 1  # default, can add job eligibility
    loc_match = 1 if candidate.get("location","").lower() in job.get("location","").lower() else 0
    stipend_match = 1
    if "expected_stipend" in candidate and pd.notna(job.get("salary")):
        stipend_match = 1  # customize numeric parsing from salary text

    tfidf_score = cosine_similarity(TF_VEC.transform([candidate["skills"]]),
                                   TF_TFIDF[merged_jobs.index.get_loc(job.name)])[0][0]
    emb_score = (embed_model.encode([candidate["skills"]], normalize_embeddings=True) @ job_embs[job.name].T)[0]
    return [skills_match, exp_match, loc_match, stipend_match, tfidf_score, emb_score]

# ---------- Threshold ----------
THRESHOLD = 0.7

for _, cand in candidates.iterrows():
    shortlisted = []
    for idx, job in merged_jobs.iterrows():
        feats = calc_features(cand, job)
        score = sum(feats)/len(feats)  # simple average; can replace with ML
        if score >= THRESHOLD:
            shortlisted.append((job["company_name"], job["job"], score))
    shortlisted.sort(key=lambda x: x[2], reverse=True)
    print(f"\nCandidate: {cand['candidate_name']}")
    print("Shortlisted Internships:")
    for company, job_title, score in shortlisted[:5]:
        print(f"  {company} - {job_title} (Score: {score:.2f})")
