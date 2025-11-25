# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:10:15 2025

@author: Anshika Agarwal
"""
import numpy as np
import pickle
loaded_model = pickle.load(open("C:/Users/Anshika Agarwal/Documents/Diabetics Prediction/trained_model.sav", "rb"))
input_data=[4,110,92,0,0,37.6,0.191,30]
input_data_np_array=np.asarray(input_data)
input_data_reshape=input_data_np_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshape)
print(prediction)
if(prediction[0]==0):
  print("The person is not diabetics")
else:
  print("The person is diabetics")