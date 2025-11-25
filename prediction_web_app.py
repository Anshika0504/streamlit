# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:06:16 2025

@author: Anshika Agarwal
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open("C:/Users/Anshika Agarwal/Documents/Diabetics Prediction/trained_model.sav", "rb"))

#creating afunction for prediction

def diabetics_prediction(input_data):
    input_data_np_array=np.asarray(input_data)
    input_data_reshape=input_data_np_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    print(prediction)
    if(prediction[0]==0):
      return "The person is not diabetics"
    else:
      return "The person is diabetics"


def main():
    
    
    #giving the title for user interface
    st.title("Diabetice prediction web app")
    
    #getting the input data from the users
    Pregnancies=st.text_input('No Of Pregnancies')
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("Blood Pressure Level")
    SkinThickness=st.text_input("Skin Thickness")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI")
    DiabetesPedigreeFunction=st.text_input("Diabetics Pedigree Function")
    Age=st.text_input("Enter the age of the person")
    
    #code for prediction
    diagnosis=''
    
    #creating the button for prediction
    if st.button("Diabetics Test Result"):
        diagnosis=diabetics_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
        
    
    