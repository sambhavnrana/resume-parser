from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import docx2txt
from nltk.tokenize import WhitespaceTokenizer
import pickle
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio 

app = Flask(__name__)

class JobPredictor:
    def __init__(self) -> None:
        save_label_encoder = open("le.pickle", "rb")
        self.le = pickle.load(save_label_encoder)
        save_label_encoder.close()

        save_classifier = open("clf.pickle", "rb")
        self.clf = pickle.load(save_classifier)
        save_classifier.close()

        # Load the TfidfVectorizer along with the vocabulary used during training
        self.tfidf_vectorizer = pickle.load(open("word_vectorizer.pickle", "rb"))

    def predict(self, resume):
        # Ensure that the input resume is transformed using the same TfidfVectorizer as during training
        resume_tfidf = self.tfidf_vectorizer.transform([resume])
        predicted = self.clf.predict(resume_tfidf)
        resume_position = self.le.inverse_transform(predicted)[0]
        return resume_position

    def predict_proba(self, resume):
        # Ensure that the input resume is transformed using the same TfidfVectorizer as during training
        resume_tfidf = self.tfidf_vectorizer.transform([resume])
        predicted_prob = self.clf.predict_proba(resume_tfidf)
        return predicted_prob[0]

text_tokenizer = WhitespaceTokenizer()
remove_characters = str.maketrans("", "", "±§!@#$%^&*()-_=+[]}{;'\:,./<>?|")
# Use the same vocabulary size as used during training
tfidf_vectorizer = TfidfVectorizer(tokenizer=text_tokenizer.tokenize, max_features=1500)  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    job_description = docx2txt.process(request.files['jd'])
    resume = docx2txt.process(request.files['resume'])

    job_predictor = JobPredictor()

    text_docx = [resume, job_description]
    
    # Ensure that the input text is transformed using the same TfidfVectorizer as during training
    tfidf_vectors = job_predictor.tfidf_vectorizer.transform(text_docx)

    similarity_score_docx = cosine_similarity(tfidf_vectors)
    match_percentage_docx = round((similarity_score_docx[0][1] * 100), 2)

    resume_position = job_predictor.predict(resume)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_percentage_docx,  # Use the match percentage out of 100
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match with JD"}
    ))
    graph = fig.to_html(full_html=False)

    chart_data = pd.DataFrame({
        "position": [cl for cl in job_predictor.le.classes_],
        "match": job_predictor.predict_proba(resume)
    })
    chart = px.bar(chart_data, x="position", y="match", title=f'Resume matched to: {resume_position}').to_html(full_html=False)

    return render_template('result.html', resume_position=resume_position, match_percentage=match_percentage_docx, graph=graph, chart=chart)

if __name__ == '__main__':
    app.run(debug=True)
