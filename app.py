import keras
import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv('scaling_data.csv')

#Loading up the Regression model we created
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#Caching the model for faster loading
@st.cache

def predict(gender, married, age, residence, glucose_level, height, weight, work_status, disease, smoker):
    #Predicting the price of the carat
    gender = int(gender == 'Female')
    married = int(married == 'Yes')
    age = (df['Mean'][0] - age)/df['Mean'][0]
    residence = int(residence == 'Urban')
    glucose_level = (df['Mean'][1] - glucose_level)/df['Mean'][1]
    bmi = weight/((height/100)**2)
    bmi = (df['Mean'][2] - bmi)/df['Mean'][2]
    no_work = int(work_status == 'Not employed')
    self_employed = int(work_status == 'Self-employed')
    disease = int(disease == 'Yes')
    smoker = int(smoker == 'Yes')
    regular_job = int(work_status == 'Private job' | work_status == 'Government job')
    prediction = loaded_model.predict(np.array([[gender, married, age, residence, glucose_level, bmi, no_work, self_employed, disease, smoker, regular_job]]))
    return prediction


st.title('Stroke Chance Predictor')
st.image('https://thumbs.dreamstime.com/b/human-heart-12427347.jpg')
st.header('Fill the details :')

gender = st.selectbox('Gender :', ['Female', 'Male', 'Other'])
married = st.selectbox('Have you ever married?', ['Yes', 'No'])
age = st.number_input('Age :', min_value = 0.3, max_value = 80.0, value = 25.0)
residence = st.selectbox('Residence type :', ['Urban', 'Rural'])
glucose_level = st.number_input('Average Glucose Level(in mg/dL) :', min_value = 10.0, max_value = 400.0, value = 120.0)
height = st.number_input('Height(in cm) :', min_value = 30.0, max_value = 220.0, value = 160.0)
weight = st.number_input('Weight(in kg) :', min_value = 5.0, max_value = 120.0, value = 70.0)
work_status = st.selectbox('Work type :', ['Private job', 'Government job', 'Self-employed', 'Not employed'])
disease = st.selectbox('Hypertension/Heart disease :', ['Yes', 'No'])
smoker = st.selectbox('Have you ever smoked?', ['Yes', 'No'])

if st.button('Predict chance of having a stroke'):
    chance = predict(gender, married, age, residence, glucose_level, height, weight, work_status, disease, smoker)
    st.success(f'The predicted chance of stroke is {chance[0]*100:.2f}%')
