import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Gemini Setup
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyAQRo9WP0QQviM31llfuglG0bL-yChu_kI")  # 🔑 Replace with your real key
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ===============================
# 1. Load Data
# ===============================
students = pd.read_csv("candidates.csv")
internships = pd.read_csv("internshipshala.csv")

# Rename internships columns to be consistent
internships.rename(columns={
    "job": "role_desc",
    "salary": "stipend",
    "duration": "time_frame",
    "desc": "skills_needed"
}, inplace=True)

internships.fillna("", inplace=True)
students.fillna("", inplace=True)

# ===============================
# 2. Preprocess Skills
# ===============================
def clean_text_skills(text):
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9, ]", " ", text)
    return " ".join(text.split())

# Clean columns
students["skills"] = students["skills"].astype(str).apply(clean_text_skills)
internships["skills_needed"] = internships["skills_needed"].astype(str).apply(clean_text_skills)

# ===============================
# 3. TF-IDF Recommendation
# ===============================
vectorizer = TfidfVectorizer(stop_words="english")
internship_tfidf = vectorizer.fit_transform(internships["skills_needed"].astype(str))

def recommend_for_student(student):
    student_skills = clean_text_skills(student["skills"])
    student_vector = vectorizer.transform([student_skills])

    similarity_scores = cosine_similarity(student_vector, internship_tfidf).flatten()

    # Add similarity to internships
    internships_copy = internships.copy()
    internships_copy["score"] = similarity_scores

    # Fallback: add skill overlap (avoids flat 0.0 issue)
    def skill_overlap(row):
        student_set = set(student_skills.split())
        job_set = set(row["skills_needed"].split())
        return len(student_set & job_set) / max(1, len(student_set))

    internships_copy["score"] = internships_copy.apply(
        lambda x: max(x["score"], skill_overlap(x)), axis=1
    )

    # stipend filtering
    if "expected_stipend" in student and not pd.isna(student["expected_stipend"]):
        try:
            expected = int(student["expected_stipend"])
            internships_copy["stipend_numeric"] = internships_copy["stipend"].astype(str)\
                .replace('[^0-9]', '', regex=True).replace('', '0').astype(int)
            internships_copy["score"] = internships_copy.apply(
                lambda x: x["score"] * 1.2 if x["stipend_numeric"] >= expected else x["score"] * 0.8,
                axis=1
            )
        except:
            pass

    return internships_copy.sort_values("score", ascending=False).head(5)

# ===============================
# 4. Classification Model
# ===============================
pairs = []
for _, student in students.iterrows():
    for _, internship in internships.iterrows():
        pairs.append({
            "student": student["candidate_name"],
            "skills": student["skills"],
            "company": internship["company_name"],
            "role": internship["role_desc"],
            "stipend": internship["stipend"],
            "required_skills": internship["skills_needed"],
            "match": int(any(skill.strip().lower() in internship["skills_needed"].lower() 
                             for skill in student["skills"].split(",")))
        })

pairs_df = pd.DataFrame(pairs)

# Encode role
le_role = LabelEncoder()
pairs_df["role_enc"] = le_role.fit_transform(pairs_df["role"].astype(str))

# Skill overlap feature
pairs_df["skill_overlap"] = pairs_df.apply(
    lambda x: len(set([s.strip().lower() for s in x["skills"].split(",")]) & 
                set([s.strip().lower() for s in x["required_skills"].split(",")])), axis=1
)

X = pairs_df[["skill_overlap", "role_enc"]]
y = pairs_df["match"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"✅ Classification Accuracy: {clf.score(X_test, y_test):.2f}")

pairs_df["probability"] = clf.predict_proba(X)[:, 1]

def classify_threshold(p):
    if p >= 0.7:
        return "Good Match"
    elif p >= 0.4:
        return "Possible Match"
    else:
        return "No Match"

pairs_df["predicted_label"] = pairs_df["probability"].apply(classify_threshold)

# ===============================
# 5. Gemini Reasoning
# ===============================
def explain_with_gemini(student, job):
    if not GEMINI_AVAILABLE:
        return "Gemini not available"
    prompt = f"""
    Candidate: {student['candidate_name']} ({student['skills']})
    Internship: {job['role_desc']} at {job['company_name']} ({job['stipend']})
    Why is this a good match? Answer in 2 sentences.
    """
    try:
        resp = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        return f"Gemini error: {e}"

# ===============================
# 6. User Input
# ===============================
print("Enter your details:")
u_name = input("Name: ")
u_skills = input("Skills (comma separated): ")
u_year = input("Year of passing: ")
u_location = input("Location: ")
u_remote = input("Remote? (yes/no): ")

user_profile = {
    "candidate_name": u_name,
    "skills": u_skills,
    "year": u_year,
    "location": u_location,
    "expected_stipend": None if u_remote.lower() not in ["yes", "y"] else 0
}

top_matches = recommend_for_student(user_profile)

results = []
for _, job in top_matches.iterrows():
    reason = explain_with_gemini(user_profile, job)
    results.append({
        "candidate": u_name,
        "skills": u_skills,
        "company": job["company_name"],
        "role": job["role_desc"],
        "stipend": job["stipend"],
        "match_score": round(job["score"], 3),
        "gemini_reason": reason
    })

# ===============================
# 7. Save Output
# ===============================
pd.DataFrame(results).to_csv("user_matches.csv", index=False)
with open("user_matches.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Matches saved in user_matches.csv & user_matches.json")
