import pandas as pd
import random

# Sample data for generating random companies, roles, skills
company_names = [
    "TechNova", "InnoSoft", "DataPulse", "WebWorks", "CloudCore", "AI Solutions",
    "NextGen Labs", "CyberSphere", "PixelEdge", "QuantumTech", "BlueWave", "SmartLogic"
]

roles = [
    "Software Developer Intern", "Data Analyst Intern", "Machine Learning Intern",
    "Frontend Developer Intern", "Backend Developer Intern", "UI/UX Designer Intern",
    "Digital Marketing Intern", "Product Management Intern", "Cybersecurity Intern"
]

skills_needed = [
    "Python", "JavaScript", "React", "Node.js", "SQL", "Data Analysis", "Machine Learning",
    "Deep Learning", "UI/UX Design", "Adobe Photoshop", "Marketing", "Communication Skills",
    "Cybersecurity Fundamentals"
]

descriptions = [
    "Assist in developing software solutions.", "Work on data analysis and reporting.",
    "Build machine learning models and pipelines.", "Design and implement web interfaces.",
    "Maintain backend services and databases.", "Create user-friendly designs.",
    "Support marketing campaigns and social media.", "Help in product planning and execution.",
    "Monitor and secure IT systems."
]

# Generate 2000 internship entries
data = []
for i in range(2000):
    company = random.choice(company_names) + f"_{random.randint(1,1000)}"
    role = random.choice(roles)
    stipend = f"₹{random.randint(5000, 50000)} / month"
    timeframe = f"{random.randint(1,6)} Months"
    desc = random.choice(descriptions)
    skills = ", ".join(random.sample(skills_needed, k=random.randint(2,5)))
    
    data.append([company, stipend, role, timeframe, desc, skills])

# Create DataFrame
df = pd.DataFrame(data, columns=["Company Name", "Stipend", "Role", "Timeframe", "Description", "Skills Needed"])

# Save to CSV
df.to_csv("internshipsssssssssssss.csv", index=False)

print("internships.csv generated successfully!")