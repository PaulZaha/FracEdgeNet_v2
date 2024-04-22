import os
import pandas as pd
import numpy as np



#Dataframe einlesen
os.chdir(os.path.join(os.getcwd(),'Dataset','Fracatlas'))
dataframe = pd.read_csv('dataset.csv')
os.chdir(os.path.join(os.getcwd(),'..'))

#Nur image_id und fractured als Spalten drinlassen
dataframe = dataframe[['image_id', 'fractured']].assign(fractured=dataframe['fractured'].astype(str))

#Alle Fractures zoomen und damit verdoppeln
fractures = dataframe[dataframe['fractured'] == "1"].copy()
fractures['image_id'] = fractures['image_id'].str[:-4] + '_zoomed.jpg'
dataframe = pd.concat([dataframe,fractures],ignore_index=True)

#Alle Fractures horizontal flippen und damit vervierfachen
fractures = dataframe[dataframe['fractured'] == "1"].copy()
fractures['image_id'] = fractures['image_id'].str[:-4] + '_flipped.jpg'
dataframe = pd.concat([dataframe,fractures],ignore_index=True)

#Rauslöschen der Dateien aus dem Dataframe der downgesampleten non-fractures
image_path = 'K:/FracEdgeNet_v2/Dataset/FracAtlas/images/full_augmented'
folder_data = os.listdir(image_path)

dataframe = dataframe[dataframe['image_id'].isin(folder_data)].copy()

dataframe.to_csv('K:/FracEdgeNet_v2/Dataset/FracAtlas/dataset_preaugmented.csv',index=False)

print(dataframe)
