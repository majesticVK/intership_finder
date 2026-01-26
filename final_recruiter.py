import pandas as pd
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
import faiss

# ---------- CONFIG ----------
CANDIDATES_CSV = "candidates_2000.csv"
OUTPUT_JSON = "matched_candidates.json"
OUTPUT_CSV = "matched_candidates.csv"
TOP_K = 10
USE_GEMINI = False  # Set True if you have Gemini API

# ---------- VACANCY INPUT FROM USER ----------
print("Enter vacancy details for shortlisting candidates:")

title = input("Job Title: ").strip()
skills_required = input("Skills required (comma-separated): ").strip().split(",")
skills_required = [s.strip() for s in skills_required if s.strip()]
location = input("Location: ").strip()
remote_input = input("Remote? (yes/no): ").strip().lower()
remote = True if remote_input in ["yes", "y"] else False
stipend = input("Stipend (e.g., 20k/month): ").strip()
description = input("Job Description: ").strip()

vacancy = {
    "title": title,
    "skills_required": skills_required,
    "location": location,
    "remote": remote,
    "stipend": stipend,
    "description": description
}

print("\nVacancy received:")
print(vacancy)

# ---------- LOAD CANDIDATES ----------
df = pd.read_csv(CANDIDATES_CSV)
df.fillna("", inplace=True)

# Combine relevant fields into one text column for matching
df["profile_text"] = (
    "Skills: " + df["skills"].astype(str) +
    ". Interests: " + df["interests"].astype(str) +
    ". Projects: " + df["projects"].astype(str) +
    ". Experience: " + df["experience_months"].astype(str) +
    " months. Preferred domains: " + df["preferred_sector_domains"].astype(str)
)

# ---------- PREPROCESSING ----------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"https?:\/\/\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return text.strip()

df["profile_text_proc"] = df["profile_text"].map(preprocess)
vacancy_text = preprocess(vacancy["title"] + " " + vacancy["description"] + " " + " ".join(vacancy["skills_required"]))

# ---------- EMBEDDINGS & FAISS ----------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
profiles = df["profile_text_proc"].tolist()
profile_embeddings = model.encode(profiles, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
dim = profile_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(profile_embeddings)

vacancy_emb = model.encode([vacancy_text], convert_to_numpy=True, normalize_embeddings=True)
D, I = index.search(vacancy_emb, TOP_K)

# ---------- PREPARE RESULTS ----------
matched_candidates = []
for score, idx in zip(D[0], I[0]):
    candidate = df.iloc[idx].to_dict()
    candidate["match_score"] = float(score)
    
    # Optional: Gemini reasoning placeholder
    if USE_GEMINI:
        candidate["reason"] = "Gemini reasoning goes here."
    else:
        candidate["reason"] = "Semantic embedding similarity match."
    
    matched_candidates.append(candidate)

# ---------- SAVE OUTPUT ----------
# JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(matched_candidates, f, ensure_ascii=False, indent=2)

# CSV
pd.DataFrame(matched_candidates).to_csv(OUTPUT_CSV, index=False)

print(f"Top {TOP_K} matched candidates saved to {OUTPUT_JSON} and {OUTPUT_CSV}")