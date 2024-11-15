from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import requests
import os
import json
import random
import pickle
import pandas as pd
import numpy as np
import re


app = Flask(__name__)
app.secret_key = os.urandom(24)

def slugify(text):
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[^a-zA-Z0-9\-]', '', text)
    return text.lower()

# chatbot handler
# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HUGGINGFACE_API_KEY = "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"

PRIOR_PROMPT = (
    "You are a friendly chatbot acting as medical consultation assistant. Answer the following questions with detailed, relevant advice. "
    "Make sure to keep answers accessible and informative for the user. "
    "Don't make up answer, if you don't know or not sure, just say so and recommend the user to seek professional help. "
    "If i greet you without asking question, just greet back without generating anything"
)

def query_huggingface(user_input):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    # Include the prior prompt and ensure all tokens are generated without truncation
    payload = {
        "inputs": f"{PRIOR_PROMPT}\n\nUser: {user_input}\n\nAssistant:",
        "parameters": {
            "max_new_tokens": 1024,  # Increase token limit to prevent truncation
            "return_full_text": False  # Only return the generated assistant response
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    # Check if the response was successful
    if response.status_code == 200:
        try:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Sorry, I couldn't process that.")
            else:
                return "Sorry, I couldn't process that."
        except requests.exceptions.JSONDecodeError:
            print("Error: Response content is not JSON:", response.text)
            return "Unable to parse response from Hugging Face API"
    else:
        # Log error details if the request was unsuccessful
        print(f"Error {response.status_code}: {response.text}")
        return f"Request failed with status code {response.status_code}"

@app.route('/chatbot-konsultasi', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.json.get("message")
        bot_response = query_huggingface(user_input)
        return jsonify({"response": bot_response})
    else:
        return render_template('pages/chatbot-konsultasi.html')
    
# Load article
def load_article():
    with open('static/data/data-penyakit.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for article in data['articles']:
            article['slug'] = slugify(article['Name'])
        return data

data = load_article()

# Load medicine
def load_medicine():
    with open('static/data/data-obat.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data['items']:
            item['slug'] = slugify(item['Nama'])
        return data

obat = load_medicine()

@app.route('/medicine/<slug>')
def medicine(slug):
    obat = load_medicine()
    medicine_data = next((item for item in obat['items'] if item["slug"] == slug), None)

    if medicine_data:
        recommended_medicine = [item for item in obat['items'] if item["slug"] != slug]
        return render_template('pages/obat-detail.html', medicines=medicine_data, recommend=recommended_medicine)
    else:
        return "Obat tidak ditemukan", 404
    
@app.route('/article/<slug>')
def article(slug):
    data = load_article()
    obat = load_medicine()
    article_data = next((item for item in data['articles'] if item["slug"] == slug), None)

    if article_data:
        
        recommended_articles = [item for item in data['articles'] if item["slug"] != slug]
        selected_articles = random.sample(recommended_articles, 3) if len(recommended_articles) >= 3 else recommended_articles
    
        filtered_medicines = [medicine for medicine in obat['items'] if medicine['Id_Penyakit'] == article_data['Id_Penyakit']]
        return render_template('pages/artikel-detail.html', articles=article_data, recommend=selected_articles, medicines=filtered_medicines)
    else:
        return "Article tidak ditemukan", 404
    
@app.route('/search-artikel.html')
def search_articles():
    data = load_article()
    search_query = request.args.get('search_query', '').lower()
    # Jika search_query kosong, tampilkan semua artikel
    if not search_query or search_query == '':
        data = load_article()

        filtered_articles = data['articles']
    else:
        filtered_articles = [article for article in data['articles'] if search_query in article['Name'].lower()]
    return render_template('pages/artikel-penyakit.html', articles=filtered_articles, search_query=search_query, new=data['articles'])

@app.route('/search-obat.html')
def search_medicines():
    obat = load_medicine()

    pencarian = request.args.get('pencarian', '').lower()
    # Jika pencarian kosong, tampilkan semua obat
    if not pencarian or pencarian == '':
        obat = load_medicine()

        filtered_medicines = obat['items']
    else:
        filtered_medicines = [item for item in obat['items'] if pencarian in item['Nama'].lower()]
    return render_template('pages/artikel-penyakit.html', medicines=filtered_medicines, pencarian=pencarian)


    
@app.route('/artikel-penyakit')
def artikel_penyakit():
    data = load_article()
    obat = load_medicine()

    section = request.args.get('section', 'articles')
    search_query = request.args.get('search_query', '').lower()
    pencarian = request.args.get('pencarian', '').lower()
    
    if section == 'articles':
        data = load_article()
        if search_query:
            filtered_articles = [article for article in data['articles'] if search_query in article['Name'].lower()]
        else:
            filtered_articles = data['articles']
        return render_template('pages/artikel-penyakit.html', articles=filtered_articles, section=section, search_query=search_query)

    elif section == 'medicines':
        obat = load_medicine()
        if pencarian:
            filtered_medicines = [item for item in obat['items'] if pencarian in item['Nama'].lower()]
        else:
            filtered_medicines = obat['items']
        return render_template('pages/artikel-penyakit.html', medicines=filtered_medicines, section=section, pencarian=pencarian)


@app.route('/rekomendasi-pengobatan')
def rekomendasi():
    return render_template('pages/rekomendasi-pengobatan.html')

@app.route('/HealthBot')
def wawasan():
    return render_template('pages/wawasan-dan-tips.html')  


# Rekomendasi

# Load the model and data files
with open('data/Source/Source/Apps/RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/Source/Source/Apps/diseases_list_indo.pkl', 'rb') as f:
    diseases_list = pickle.load(f)
with open('data/Source/Source/Apps/symptoms_dict_indo.pkl', 'rb') as f:
    symptoms_dict = pickle.load(f)

description = pd.read_csv('data/Source/Source/Data/new_desc.csv')
precautions_df = pd.read_csv('data/Source/Source/Data/new_pre.csv')
medications = pd.read_csv('data/Source/Source/Data/obat.csv')
diets = pd.read_csv('data/Source/Source/Data/fix_diets.csv')
workout_df = pd.read_csv('data/Source/Source/Data/workout.csv')

# Helper function to process data
def clean_text(data):
    if isinstance(data, list):
        return '\n'.join([str(item).strip().replace("'", "").replace("[", "").replace("]", "") for item in data])
    return str(data).strip().replace("'", "").replace("[", "").replace("]", "")

def helper(dis):
    # Fetch and clean data
    desc = clean_text(description[description['Penyakit'] == dis]['Keterangan'].values[0])
    
    # Convert the precautions into a list
    pre = [clean_text(pre) for pre in precautions_df[precautions_df['Penyakit'] == dis][['Tindakan pencegahan_1', 'Tindakan pencegahan_2', 'Tindakan pencegahan_3', 'Tindakan pencegahan_4']].values.flatten()]
    
    # Fetch medications, diet, and workout data
    med = clean_text(medications[medications['Penyakit'] == dis]['Pengobatan'].values)
    die = clean_text(diets[diets['Penyakit'] == dis]['Diet'].values)
    wrkout = clean_text(workout_df[workout_df['penyakit'] == dis]['olahraga'].values)
    
    # Split comma-separated strings into lists if needed
    med_list = [item.strip() for item in med.split(',')] if isinstance(med, str) else []
    diet_list = [item.strip() for item in die.split(',')] if isinstance(die, str) else []
    workout_list = [item.strip() for item in wrkout.split(',')] if isinstance(wrkout, str) else []

    # Split workout text on '\n' to create a list
    workout_list1 = workout_list[0].split('\n') if isinstance(wrkout, str) else []
    
    return desc, pre, med_list, diet_list, workout_list1


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[model.predict([input_vector])[0]]

# Flask routes

@app.route('/', methods=['GET', 'POST'])
def home():
    data = load_article()
    articles=data['articles']
    selected_articles = random.sample(articles, 2) if len(articles) >= 2 else articles
    
    return render_template('home.html', symptoms=list(symptoms_dict.keys()), articles=selected_articles)

# Recommendation route for displaying prediction results
@app.route('/recommendation', methods=['GET'])
def recommendation():
    symptoms = request.args.get('symptoms', '').split(',')
    predicted_disease = get_predicted_value(symptoms)
    desc, pre, med_list, diet_list, workout_list = helper(predicted_disease)

    data = load_article()
    articles=data['articles']
    selected_articles = random.sample(articles, 2) if len(articles) >= 2 else articles
    
    return render_template(
        'pages/rekomendasi-pengobatan.html',
        disease=predicted_disease,
        description=desc,
        precautions=pre,
        medications=med_list,
        workout=workout_list,
        diet=diet_list,
        articles=selected_articles
    )
    
if __name__ == '__main__':
    app.run(debug=True)