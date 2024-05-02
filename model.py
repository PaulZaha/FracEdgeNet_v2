import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import keras
import os
from keras.layers import Input, Add, Conv2D, GlobalAveragePooling2D, Reshape, Layer, Dense
from keras.models import Model

from model_v2 import EfficientNetV2S

#from keras.applications import EfficientNetV2S, EfficientNetV2M

from model_utils import *

def EfficientNetB4(train_generator,validation_generator,test_generator):
    
    input_layer = layers.Input(shape=(380,380,3))

    model_EfficientNetB4 = tf.keras.applications.EfficientNetB4(weights='imagenet',input_tensor = input_layer,include_top = False)

    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')

    model = tf.keras.models.Sequential([
        model_EfficientNetB4,
        flatten,
        classifier
    ])
    #Layer untrainable machen
    for layer in model.layers[:-1]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
        layer.trainable=False
        
    model_compiler(model)

    print("Ab hier: Classifier Fitting")
    model_fitter(model,train_generator,validation_generator,2)

    model_evaluater(test_generator)






#!Note: TF 2.16.1 für EfficientNetV2 benötigt, allerdings läuft da auf Linux noch kein GPU Support.





def FracEdgeNet(train_generator,validation_generator,test_generator):
    #create_model((256,256,8))
    input_layer = layers.Input(shape=(380,380,3))

    EffV2S = tf.keras.applications.EfficientNetV2S(weights='imagenet',input_tensor = input_layer,include_top = False)
    pretrained_weights = EffV2S.get_weights()

    
    
    FracEdgeNet = EfficientNetV2S(include_top=False,weights=None, input_tensor=input_layer)
    #!Ab hier: Neues modell zusammenschustern


    # pretrained_weights.insert(1, np.array([]))
    # pretrained_weights.insert(1, np.array([]))
    # pretrained_weights.insert(1, np.array([]))
    # pretrained_weights.insert(1, np.array([]))
    # pretrained_weights.insert(1, np.array([]))
    # pretrained_weights.insert(1, np.array([]))

    # #Für Classifier am Ende
    # pretrained_weights.append(np.array([]))
    # pretrained_weights.append(np.array([]))


    for layer in FracEdgeNet.layers:
        try:
            # Suche den entsprechenden Layer in den vortrainierten Gewichten
            pretrained_layer = EffV2S.get_layer(layer.name)
            # Setze die Gewichte des aktuellen Layers auf die Gewichte des entsprechenden Layers in den vortrainierten Gewichten
            layer.set_weights(pretrained_layer.get_weights())
        except ValueError as e:
            print("ValueError in layer:", layer.name)
            print("Exception message:", e)





    #FracEdgeNet.summary()





    for layer in FracEdgeNet.layers[:-2]:
        layer.trainable=False
    #FracEdgeNet.save('trained.h5')
    #FracEdgeNet.save('structure.h5')
        
    model_compiler(FracEdgeNet)

    print("Ab hier: Classifier Fitting")
    model_fitter(FracEdgeNet,train_generator,validation_generator,3)

    model_evaluater(test_generator,FracEdgeNet)




def create_kirsch_filters():
    kirsch_filters = [
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],  # Kirsch-Nord
        [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],  # Kirsch-Nordost
        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],  # Kirsch-Ost
        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],  # Kirsch-Südost
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],  # Kirsch-Süd
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],  # Kirsch-Südwest
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],  # Kirsch-West
        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]   # Kirsch-Nordwest
    ]
    return kirsch_filters
    #return np.array(kirsch_filters, dtype=np.float32)


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    conv_layer = Conv2D(1, (3, 3), padding='same', activation='relu')(inputs)  # Faltungsschicht
    kirsch_filters = create_kirsch_filters()
    conv_layer = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer=tf.constant_initializer(np.array(create_kirsch_filters())))(conv_layer)  # Faltung mit Kirsch-Matrizen
    reduc = GlobalAveragePooling2D()(conv_layer)
    
    conv_layer = Conv2D(1, (1,1), padding='same', activation='relu')(reduc[:, tf.newaxis, tf.newaxis])
    conv_layer = Add()([conv_layer, inputs])  # Skip Connection
    model = Model(inputs=inputs, outputs=conv_layer)
    return model




def main():
    pass

if __name__ == "__main__":
    main()