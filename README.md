# AI Internship Recommender System

## Smart internship matching using Hybrid AI (ML + NLP + Semantic Search)

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue"> <img src="https://img.shields.io/badge/ML-RandomForest-green"> <img src="https://img.shields.io/badge/NLP-TFIDF-orange"> <img src="https://img.shields.io/badge/Embeddings-FAISS-purple"> <img src="https://img.shields.io/badge/Status-Active-success"> </p>
 

## An AI-powered system that recommends internships based on a candidate’s profile using:

TF-IDF similarity
Semantic embeddings (Sentence Transformers + FAISS)
Machine Learning (Random Forest)
Optional LLM reasoning (Gemini)

Built as a hybrid pipeline combining rule-based + ML + semantic matching.

## ✨ Key Features

✔️ Hybrid Recommendation Engine
Combines TF-IDF + embeddings + ML scoring

✔️ Resume Input Support
Upload .txt resume for better matching

✔️ Semantic Search (FAISS)
Captures deeper meaning beyond keywords

✔️ ML-Based Ranking
Learns patterns using Random Forest

✔️ Explainable Results
Optional AI explanations for recommendations

✔️ Interactive UI
Built with Streamlit for real-time results

## 🧠 System Architecture
```
User Input

   ↓

TF-IDF Similarity

   ↓
   
Embedding Similarity (FAISS)

   ↓
   
Feature Engineering

   ↓
   
Random Forest Model

   ↓
   
Final Ranked Internships
```


## 🖥️ **Project Structure**

````
AI-internship-recommender
├── app.py              # Streamlit UI
├── shortlister.py      # Hybrid scoring engine
├──  intern.py           # ML + TF-IDF logic
├──  pf.py               # Data preprocessing
├── datasets/           # CSV datasets
````

## ⚙️ Installation
```git clone https://github.com/your-username/internship-recommender.git```

## cd internship-recommender
```pip install -r requirements.txt```

## ▶️ Run the App
```streamlit run app.py```
 

## 🧪 How It Works
1. Enter skills / upload resume 
2. System extracts features
3. Runs hybrid matching:
4. TF-IDF
5. Embeddings
6. ML scoring
7. Displays top internship matches
   
## ⚠️ Limitations
1.Depends on dataset quality

2.Requires preprocessing for best results

3.API keys (Gemini) must be configured manually (if needed for fallback)


 ## 🔮 Future Improvements
 #### 🔊 Voice-based input
 #### 📄 PDF resume parsing
 #### 🌐 Deployment (Web app)
 ####🧠 Fine-tuned recommendation model
 
 ### 👤 Author

#### Vansh Kumar
ECE + AI | Building intelligent systems

## a quick demo video
#### https://youtu.be/EfanqvZfz6E
