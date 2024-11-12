import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and data
with open('RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('diseases_list_indo.pkl', 'rb') as f:
    diseases_list = pickle.load(f)

with open('symptoms_dict_indo.pkl', 'rb') as f:
    symptoms_dict = pickle.load(f)

# Load data files from specified paths
description = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/indo/new_desc.csv")
precautions_df = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/indo/new_pre.csv")
medications = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Apps/ind/Data-Indo/fix_medication.csv")
diets = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/Apps/ind/Data-Indo/fix_diets.csv")
workout_df = pd.read_csv("C:/Users/Widnyana/Downloads/NAMA SAYA/MAHASISWA_NAMA SAYA_V1/Source/indo/workout.csv")

# Helper functions
def clean_text(data):
    # Joins items in lists and removes unwanted characters like [] and ''
    if isinstance(data, list):
        return '\n'.join([str(item).strip().replace("'", "").replace("[", "").replace("]", "") for item in data])
    return str(data).strip().replace("'", "").replace("[", "").replace("]", "")

def helper(dis):
    # Fetch and clean data
    desc = clean_text(description[description['Penyakit'] == dis]['Keterangan'].values[0])
    pre = [clean_text(pre) for pre in precautions_df[precautions_df['Penyakit'] == dis][['Tindakan pencegahan_1', 'Tindakan pencegahan_2', 'Tindakan pencegahan_3', 'Tindakan pencegahan_4']].values.flatten()]
    med = clean_text(medications[medications['Penyakit'] == dis]['Pengobatan'].values)
    die = clean_text(diets[diets['Penyakit'] == dis]['Diet'].values)
    wrkout = clean_text(workout_df[workout_df['penyakit'] == dis]['olahraga'].values)
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
        # Displaying precautions in a list format
        for idx, precaution in enumerate(pre, 1):
            st.write(f"{idx}. {precaution}")

        st.subheader("Medications")
        # Splitting the medications and displaying each item as a list
        for idx, med in enumerate(med.split(','), 1):
            st.write(f"{idx}. {med.strip()}")

        st.subheader("Workout Suggestions")
        # Displaying workout suggestions in a list format
        for idx, workout in enumerate(wrkout.split('\n'), 1):
            st.write(f"{idx}. {workout}")

        st.subheader("Diet Recommendations")
        # Splitting the diet recommendations and displaying each item as a list
        for idx, diet in enumerate(die.split(','), 1):
            st.write(f"{idx}. {diet.strip()}")
    else:
        st.write("Please select symptoms to predict the disease.")
