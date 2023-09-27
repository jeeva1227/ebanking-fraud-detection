from django.db import models
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
from keras.models import load_model
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from PIL import Image
import pandas as pd 

# Testing phase
rf = pickle.load(open("rf.pkl", 'rb'))
nb = pickle.load(open("nb.pkl", 'rb'))
bagging = pickle.load(open("bagging.pkl", 'rb'))

data = pd.read_csv('test.csv')
print("+++++++++++++++++")
print(data.head(1))


#ss =  pickle.load(open("scaler.pkl", 'rb'))
#test_data = data.drop(['PKT_CLASS'])
#test_data =   data.drop(['PKT_CLASS'], axis = 1) 

#test_data = ss.transform(data)



def predict(algo,row): 
	print(row)
	print(algo)
	filter_data = data.loc[row].values.reshape(1, -1)
	print(filter_data.shape)
	print(filter_data)
	if algo=='rf':
		y_pred= bagging.predict(filter_data)
		return y_pred[0]
	else:
		y_pred=bagging.predict(filter_data)
		return y_pred[0]

