import streamlit as st
import PyPDF2
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load Hugging Face model for classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
generator = pipeline("text-generation", model="mistralai/Mistral-7B")
labels = ["Strong", "Needs Improvement", "Rejected"]

# Sample training data for better classification
resume_texts = ["Experienced AI Engineer with Python skills", "Fresher with knowledge of ML", "Weak resume without relevant experience"]
resume_labels = ["Strong", "Needs Improvement", "Rejected"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(resume_texts)
clf = RandomForestClassifier()
clf.fit(X_train, resume_labels)

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def analyze_resume(resume_text):
    X_test = vectorizer.transform([resume_text])
    category = clf.predict(X_test)[0]
    return category

def get_resume_feedback(resume_text):
    prompt = f"Analyze this resume and provide feedback with three sections:\n1. Strengths\n2. Areas to Improve\n3. Suggested Skills: {resume_text}"
    response = generator(prompt, max_length=500)
    return response[0]['generated_text']

def extract_candidate_details(resume_text):
    name_match = re.search(r"Name[:\s]+([A-Za-z\s]+)", resume_text)
    name = name_match.group(1) if name_match else "Not Found"
    return name

# Streamlit UI
st.title("AI-Powered Resume Analyzer 🚀 (Free & Open Source)")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    candidate_name = extract_candidate_details(resume_text)
    category = analyze_resume(resume_text)
    feedback = get_resume_feedback(resume_text)
    score = np.random.uniform(60, 95)  # Generating a random score for now

    st.subheader("Candidate Details:")
    st.write(f"**Name:** {candidate_name}")
    
    st.subheader("Resume Category:")
    st.write(f"📝 {category}")
    
    st.subheader("Resume Score:")
    st.progress(score / 100)
    st.write(f"**{score:.2f}%**")
    
    st.subheader("AI Feedback:")
    st.write(feedback)
    
    st.download_button("Download Feedback", feedback, file_name="resume_feedback.txt")

# Run the app: streamlit run app.py
