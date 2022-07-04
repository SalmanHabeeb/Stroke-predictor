#Import necessary packages
import keras
import streamlit as st
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('scaling_data.csv')                             #Read the scaling data from csv file
json_file = open('model.json', 'r')                              #Open the file containing model architecture
loaded_model_json = json_file.read()                             #Read the json content into the variable
json_file.close()                                                #Close the json file
loaded_model = keras.models.model_from_json(loaded_model_json)   #Read the model architecture from metadata
loaded_model.load_weights("model.h5")                            #Load weights from the h5 file

#Caching the model for faster loading
@st.cache

def predict(gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, height,  weight, smoking_status,):
    '''
        This function takes text values as input and converts them into numerical values which the model can understand.
        It returns the predicted chance.
    '''
    gender = int(gender == 'Female')
    age = (age - df['Mean'][0])/df['Mean'][0]
    hypertension = int(hypertension == 'Yes')
    heart_disease = int(heart_disease == 'Yes')
    married = int(married == 'Yes')
    if (work_status == 'Private job') | (work_status == 'Government job'):
        work_status = 0.5
    elif work_status == 'Self-employed':
        work_status = 1.0
    elif work_status == 'Not employed':
        work_status = 0.0 
    residence = int(residence == 'Urban')
    glucose_level = (glucose_level - df['Mean'][1])/df['Mean'][1]
    bmi = weight/((height/100)**2)
    bmi = (bmi - df['Mean'][2])/df['Mean'][2]
    smoking_status = int(smoking_status == 'Yes')
    prediction = loaded_model.predict(np.array([[gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, bmi, smoking_status,]]))
    return prediction


st.title('Stroke Chance Predictor')                                    #Set the title for the application. This appears at top of page.
st.image('https://thumbs.dreamstime.com/b/human-heart-12427347.jpg')   #Set the link to image which will be used as cover image for application
st.header('Fill the details :')                                        #Set the form title. This appears before input form

#Take input variables as entry
#st.selectbox() uses dropbox for input entry
#st.number_input() uses numerical input box for entry. @Caution: Use either all integer values or all float values in st.number_input(), else error is displayed.
gender = st.selectbox('Gender :', ['Female', 'Male', 'Other'])
age = st.number_input('Age :', min_value = 0.3, max_value = 80.0, value = 25.0)
hypertension = st.selectbox('Hypertension :', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease :', ['No', 'Yes'])
married = st.selectbox('Have you ever married?', ['No', 'Yes'])
work_status = st.selectbox('Work type :', ['Private job', 'Government job', 'Self-employed', 'Not employed'])
residence = st.selectbox('Residence type :', ['Urban', 'Rural'])
glucose_level = st.number_input('Average Glucose Level(in mg/dL) :', min_value = 10.0, max_value = 400.0, value = 120.0)
height = st.number_input('Height(in cm) :', min_value = 30.0, max_value = 220.0, value = 160.0)
weight = st.number_input('Weight(in kg) :', min_value = 5.0, max_value = 120.0, value = 70.0)
smoking_status = st.selectbox('Have you ever smoked?', ['No', 'Yes'])

#If button is pressed, display the predicted value
if st.button('Predict chance of having a stroke'):
    chance = predict(gender, age, hypertension, heart_disease, married, work_status, residence, glucose_level, height,  weight, smoking_status,)
    st.success(f'The predicted chance of stroke is {chance[0][0]*100:.2f}%')
