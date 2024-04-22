#region imports and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import tensorflow as tf
import xml.etree.ElementTree as ET

from keras.preprocessing.image import ImageDataGenerator

import sklearn
from sklearn.model_selection import KFold, train_test_split

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import *


def create_generators(train_df,test_df,targetsize):
    """
    Creates train_, validation_, and test_generator to feed Model with batches of data
    """
    #images pathing
    path = os.path.join(os.getcwd(),'FracAtlas','images','full_augmented')
    print(os.getcwd())
    #Create DataGenerator for training and validation data with augmentation
    #Note: For EfficientNet remove rescaling
    datagen = ImageDataGenerator(#rescale = 1./255,
                                                              rotation_range=10 
                                                              ,width_shift_range=0.05
                                                              ,height_shift_range=0.05
                                                              ,validation_split=0.2)
    
    #Create DataGenerator for testing without augmentation
    #Note: For EfficientNet remove rescaling
    test_datagen = ImageDataGenerator(#rescale=1./255
        )


    train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        subset='training'
                                                        ,batch_size=16)
    

    validation_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        subset='validation'
                                                        ,batch_size=16)

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory=path,
                                                        x_col='image_id',
                                                        y_col='fractured',
                                                        class_mode='binary',
                                                        color_mode = 'rgb',
                                                        shuffle=True,
                                                        target_size=targetsize,
                                                        batch_size=1)

    return train_generator, validation_generator, test_generator


def main():
    os.chdir(os.path.join(os.getcwd(),'Dataset','FracAtlas'))
    dataframe = pd.read_csv('dataset_preaugmented.csv')
    os.chdir(os.path.join(os.getcwd(),'..'))
    
    dataframe = dataframe[['image_id', 'fractured']].assign(fractured=dataframe['fractured'].astype(str))


    original_indices = [idx for idx in range(len(dataframe)) if not dataframe['fractured'].iloc[idx].endswith('ed.jpg')]

    # Führe den Train-Test-Split basierend auf den Originalindizes durch
    train_indices, test_indices = train_test_split(original_indices, train_size=0.9, shuffle=True)

    # Wende die Indizes auf das DataFrame an, um die Trainings- und Testdatensätze zu erhalten
    train_dataset = dataframe.iloc[train_indices]
    test_dataset = dataframe.iloc[test_indices]

    #gewicht = round(((dataframe['fractured'].value_counts()).get(0,1))/(dataframe['fractured'].value_counts()).get(1,0),3)
    
    #Set targetsize
    targetsize = (380,380)
    
    #Create generators from train/test-split with chosen targetsize
    train_generator,validation_generator, test_generator = create_generators(train_dataset,test_dataset,targetsize)


    EfficientNetV2S(train_generator,validation_generator,test_generator)


if __name__ == "__main__":
    main()