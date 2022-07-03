import keras
import streamlit as st
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('scaling_data.csv')
global work_status_dict
work_status_dict = pickle.load('work_status_dict')

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

def predict(gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, height,  weight, smoking_status,):
    #Predicting the price of the carat
    gender = int(gender == 'Female')
    age = (age - df['Mean'][0])/df['Mean'][0]
    hypertension = int(hypertension == 'Yes')
    heart_disease = int(heart_disease == 'Yes')
    married = int(married == 'Yes')
    for key in work_status_dict:
        if key == work_status:
            work_status = work_status_dict[key]
    residence = int(residence == 'Urban')
    glucose_level = (glucose_level - df['Mean'][1])/df['Mean'][1]
    bmi = weight/((height/100)**2)
    bmi = (bmi - df['Mean'][2])/df['Mean'][2]
    smoking_status = int(smoking_status == 'Yes')
    prediction = loaded_model.predict(np.array([[gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, bmi, smoking_status,]]))
    return prediction


st.title('Stroke Chance Predictor')
st.image('https://thumbs.dreamstime.com/b/human-heart-12427347.jpg')
st.header('Fill the details :')

gender = st.selectbox('Gender :', ['Female', 'Male', 'Other'])
age = st.number_input('Age :', min_value = 0.3, max_value = 80.0, value = 25.0)
hypertension = st.selectbox('Hypertension :', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease :', ['Yes', 'No'])
married = st.selectbox('Have you ever married?', ['Yes', 'No'])
work_status = st.selectbox('Work type :', ['Private job', 'Government job', 'Self-employed', 'Not employed'])
residence = st.selectbox('Residence type :', ['Urban', 'Rural'])
glucose_level = st.number_input('Average Glucose Level(in mg/dL) :', min_value = 10.0, max_value = 400.0, value = 120.0)
height = st.number_input('Height(in cm) :', min_value = 30.0, max_value = 220.0, value = 160.0)
weight = st.number_input('Weight(in kg) :', min_value = 5.0, max_value = 120.0, value = 70.0)
smoking_status = st.selectbox('Have you ever smoked?', ['Yes', 'No'])

if st.button('Predict chance of having a stroke'):
    chance = predict(gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, height,  weight, smoking_status,)
    st.success(f'The predicted chance of stroke is {chance[0][0]*100:.2f}%')