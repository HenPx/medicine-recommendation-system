import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and data
with open('random.pkl', 'rb') as f:
    model = pickle.load(f)

with open('diseases_list.pkl', 'rb') as f:
    diseases_list = pickle.load(f)

with open('symptoms_dict.pkl', 'rb') as f:
    symptoms_dict = pickle.load(f)

# Load data files from specified paths
description = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Data/description.csv")
precautions_df = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Data/precautions_df.csv")
medications = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Data/medications.csv")
diets = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Data/diets.csv")
workout_df = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Data/workout_df.csv")

# Helper functions
def clean_text(data):
    # Joins items in lists and removes unwanted characters, if any
    if isinstance(data, list):
        return ', '.join([str(item).replace('\n', '').strip() for item in data])
    return str(data).replace('\n', '').strip()

def helper(dis):
    # Fetch and clean data
    desc = clean_text(description[description['Disease'] == dis]['Description'].values[0])
    pre = [clean_text(pre) for pre in precautions_df[precautions_df['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()]
    med = clean_text(medications[medications['Disease'] == dis]['Medication'].values)
    die = clean_text(diets[diets['Disease'] == dis]['Diet'].values)
    wrkout = clean_text(workout_df[workout_df['disease'] == dis]['workout'].values)
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[model.predict([input_vector])[0]]

# Streamlit App
st.title("Disease Prediction App")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect("Select your symptoms", options=list(symptoms_dict.keys()))

# Predict button
if st.button("Predict Disease"):
    if selected_symptoms:
        predicted_disease = get_predicted_value(selected_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        # Display results
        st.subheader("Predicted Disease")
        st.write(predicted_disease)

        st.subheader("Description")
        st.write(desc)

        st.subheader("Precautions")
        st.write(" • ".join(pre))

        st.subheader("Medications")
        st.write(med)

        st.subheader("Workout Suggestions")
        st.write(wrkout)

        st.subheader("Diet Recommendations")
        st.write(die)
    else:
        st.write("Please select symptoms to predict the disease.")
