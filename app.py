from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import random
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load article
def load_article():
    with open('menu2.json', 'r', encoding='utf-8') as file:
        return json.load(file)

data = load_article()

# Load medicine
def load_medicine():
    with open('menu3.json', 'r', encoding='utf-8') as file:
        return json.load(file)
    
obat = load_medicine()


@app.route('/medicine/<name>')
def medicine(name):
    obat = load_medicine()
    medicine_data = next((item for item in obat['items'] if item["medications"]["name"] == name), None)
    
    if medicine_data:
        recommended_medicine = [item for item in obat['items'] if item["medications"]["name"] != name]
        
        return render_template('pages/obat-detail.html', medicines=medicine_data, recommend=recommended_medicine)
    else:
        return "Obat tidak ditemukan", 404
    
@app.route('/article/<title>')
def article(title):
    data = load_article()
    obat = load_medicine()

    article_data = next((item for item in data['articles'] if item["title"] == title), None)
    

    if article_data:
        recommended_articles = [item for item in data['articles'] if item["title"] != title]
        filtered_medicines = [medicine for medicine in obat['items'] if medicine['id_penyakit'] == article_data['id']]

        return render_template('pages/artikel-detail.html', articles=article_data, recommend=recommended_articles, medicines = filtered_medicines)
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
        filtered_articles = [article for article in data['articles'] if search_query in article['title'].lower()]
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
        filtered_medicines = [item for item in obat['items'] if pencarian in item['medications']['name'].lower()]
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
            filtered_articles = [article for article in data['articles'] if search_query in article['title'].lower()]
        else:
            filtered_articles = data['articles']
        return render_template('pages/artikel-penyakit.html', articles=filtered_articles, section=section, search_query=search_query)

    elif section == 'medicines':
        obat = load_medicine()
        if pencarian:
            filtered_medicines = [item for item in obat['items'] if pencarian in item['medications']['name'].lower()]
        else:
            filtered_medicines = obat['items']
        return render_template('pages/artikel-penyakit.html', medicines=filtered_medicines, section=section, pencarian=pencarian)


@app.route('/rekomendasi-pengobatan')
def rekomendasi():
    return render_template('pages/rekomendasi-pengobatan.html')

@app.route('/wawasan-dan-tips')
def wawasan():
    return render_template('pages/wawasan-dan-tips.html')  


# Rekomendasi

# Load the model and data files
with open('data/Source/Source/Apps/random.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/Source/Source/Apps/diseases_list.pkl', 'rb') as f:
    diseases_list = pickle.load(f)
with open('data/Source/Source/Apps/symptoms_dict.pkl', 'rb') as f:
    symptoms_dict = pickle.load(f)

description = pd.read_csv('data/Source/Source/Data/description.csv')
precautions_df = pd.read_csv('data/Source/Source/Data/precautions_df.csv')
medications = pd.read_csv('data/Source/Source/Data/medications.csv')
diets = pd.read_csv('data/Source/Source/Data/diets.csv')
workout_df = pd.read_csv('data/Source/Source/Data/workout_df.csv')

# Helper function to process data
def clean_text(data):
    if isinstance(data, list):
        return ', '.join([str(item).replace('\n', '').strip() for item in data])
    return str(data).replace('\n', '').strip()

def helper(dis):
    # Retrieve and clean each data field, ensuring no empty or NaN values are shown
    desc = clean_text(description[description['Disease'] == dis]['Description'].values[0])
    
    # Filter out empty precautions
    pre = [clean_text(pre) for pre in precautions_df[precautions_df['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten() if pre]

    # Check if each field has a value before displaying
    med = clean_text(medications[medications['Disease'] == dis]['Medication'].values)
    med = med if med else 'No specific medication recommended.'
    
    die = clean_text(diets[diets['Disease'] == dis]['Diet'].values)
    die = die if die else 'No specific diet recommended.'
    
    wrkout = clean_text(workout_df[workout_df['disease'] == dis]['workout'].values)
    wrkout = wrkout if wrkout else 'No specific workout recommended.'
    
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
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
    desc, pre, med, die, wrkout = helper(predicted_disease)
    
    return render_template(
        'pages/rekomendasi-pengobatan.html',
        disease=predicted_disease,
        description=desc,
        precautions=pre,
        medications=med,
        workout=wrkout,
        diet=die
    )
    
if __name__ == '__main__':
    app.run(debug=True)