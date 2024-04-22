import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import keras
import os
from keras.layers import Input, Add, Conv2D, GlobalAveragePooling2D, Reshape
from keras.models import Model
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


def EfficientNetV2S(train_generator,validation_generator,test_generator):
    create_model((256,256,8))
    input_layer = layers.Input(shape=(373,373,3))

    EffV2S = tf.keras.applications.EfficientNetV2S(weights='imagenet',input_tensor = input_layer,include_top = False)
    V2S_adap = tf.keras.models.Sequential()

    #!Ab hier: Neues modell zusammenschustern


    for layer in EffV2S.layers[:2]:
        V2S_adap.add(layer)
    
    V2S_adap.add(create_model(input_shape=V2S_adap.layers[-1].output_shape[1:]))

    

    for layer in EffV2S.layers[2:8]:
        V2S_adap.add(layer)
    
    act1_out = V2S_adap.get_layer('stem_activation').output
    act2_out = V2S_adap.get_layer('block1a_project_activation').output

    added_output = Add()([act1_out, act2_out])

    target_layer_output = V2S_adap.get_layer('block1a_project_conv').output

    new_output = Add()([target_layer_output,added_output])


    new_model = Model(inputs=V2S_adap.input, outputs=added_output)
    new_model.summary()

    #!hier kann dann wieder draufgelayert werden













        
    V2S_adap.summary()

    for layer in EffV2S.layers:
        layer.trainable=False
    flatten = tf.keras.layers.Flatten()
    classifier = tf.keras.layers.Dense(1,activation='sigmoid')
    tf.keras.saving.save_model(EffV2S,os.path.join(os.getcwd(),'model.h5'),save_format='h5')
    model = tf.keras.models.Sequential([
        EffV2S,
        flatten,
        classifier
    ])
    model.save("test.h5")
    #Layer untrainable machen
    # for layer in model.layers[:-2]: #auf -1 ändern, wenn nur der finale classifier und keine Dense schicht
    #     layer.trainable=False
        
    model_compiler(model)

    print("Ab hier: Classifier Fitting")
    model_fitter(model,train_generator,validation_generator,10)

    model_evaluater(test_generator)




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
    return np.array(kirsch_filters, dtype=np.float32)


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    conv_layer = Conv2D(1, (3, 3), padding='same', activation='relu')(inputs)  # Faltungsschicht
    kirsch_filters = create_kirsch_filters()
    conv_layer = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer=tf.constant_initializer(kirsch_filters))(conv_layer)  # Faltung mit Kirsch-Matrizen
    reduc = GlobalAveragePooling2D()(conv_layer)
    conv_layer = Conv2D(1, (1,1), padding='same', activation='relu')(tf.expand_dims(tf.expand_dims(reduc, axis=1),axis=1))
    conv_layer = Add()([conv_layer, inputs])  # Skip Connection
    model = Model(inputs=inputs, outputs=conv_layer)
    #model.summary()
    return model
