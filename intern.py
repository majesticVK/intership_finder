import pandas as pd
import re

pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.max_rows", 100)       # show up to 100 rows (increase if needed)
pd.set_option("display.width", None)         # don't wrap lines
pd.set_option("display.colheader_justify", "left")  # align headers neatly

# Load data
internships = pd.read_csv("internshipshala.csv", encoding="utf-8")
internships.columns = internships.columns.str.strip()

formatted_jobs = pd.read_csv("formatted_jobs.csv")

# Job mapping
job_mapping = {
    "HR Generalist": "HR Specialist",
    "Field Sales": "Sales Representative",
    "Architecture": "UX Designer",
    "Fashion Design": "Graphic Designer",
    "Operations": "Project Manager",
    "Business Development (Sales)": "Sales Representative",
    "Video Editor": "Graphic Designer",
    "Sales & Communication Manager": "Marketing Manager"
}

# Apply mapping
internships["mapped_job_title"] = internships["job"].map(job_mapping)

# --- 🟢 Salary Cleaning Function ---
def clean_salary(s):
    if pd.isna(s):
        return None, None, None, None
    s = str(s).replace("â‚¹", "").replace("₹", "").replace(",", "").strip()
    
    # Extract numbers
    nums = re.findall(r"\d+", s)
    nums = list(map(int, nums))
    
    # Pay frequency
    freq = "month"
    if "week" in s.lower():
        freq = "week"
    elif "day" in s.lower():
        freq = "day"
    elif "year" in s.lower():
        freq = "year"
    
    if len(nums) == 1:
        return nums[0], nums[0], nums[0], freq
    elif len(nums) >= 2:
        mn, mx = nums[0], nums[1]
        avg = (mn + mx) // 2
        return mn, mx, avg, freq
    return None, None, None, freq

# Apply salary parsing
internships[["salary_min", "salary_max", "salary_avg", "salary_freq"]] = internships["salary"].apply(
    lambda x: pd.Series(clean_salary(x))
)

# Merge datasets
merged = internships.merge(
    formatted_jobs,
    left_on="mapped_job_title",
    right_on="job_title",
    how="left"
)

if "job_title" in merged.columns:
    merged = merged.drop(columns=["job_title"])

# ✅ Final dataset preview
print(merged[[
    "company_name", "job", "location", "salary", "salary_min", "salary_max", "salary_avg", "salary_freq",
    "duration", "Skills_required", "Industry", "Pay_grade"
]].head())
