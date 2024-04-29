import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import keras
import os
import numpy as np

#optimized learning rate for hypertuning EfficientNetB4
def model_compiler(model):
    initial_lr = 1e-4
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=130,
        decay_rate=0.94,
        staircase=True
    )
    print("scheduler jetzt da")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy','FalseNegatives','FalsePositives','Precision','Recall'])

def model_fitter(model,train_generator,validation_generator,epochs):
    history = model.fit(train_generator,validation_data=validation_generator,epochs=epochs
                        #,class_weight = {0: 1, 1: 5}
                        #,callbacks =[model_callback,stopper,tensorboard_callback,learning_rate_scheduler,custom_lr_update_callback]
                        #,steps_per_epoch=500
                        )

    print(history.history['accuracy'])
    print(history.history['loss'])
    print(history.history['val_accuracy'])
    print(history.history['val_loss'])
    
    #Create plots
    plt.figure(figsize=(12, 6))

    #Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()

def model_evaluater(test_generator):
    #Load model from best epoch
    model = tf.keras.models.load_model(os.path.join(os.getcwd(),"trained.h5"))

    #Evaluate the model and save to results
    results = model.evaluate(test_generator)

    #Create dict with evaluation metrics for testing dataset
    eval_metrics = {"Accuracy: ": results[1]
                    ,"Loss: ": results[0]
                    ,"False_Negatives: ":int(results[2])
                    ,"False_Positives: ":int(results[3])
                    ,"Precision: ":results[4]
                    ,"Recall: ":results[5]
                    }
    print(eval_metrics)


    False_Negatives = int(eval_metrics['False_Negatives: '])
    False_Positives = int(eval_metrics['False_Positives: '])

    True_Positives = int((eval_metrics['Precision: ']*False_Positives)/(1-eval_metrics['Precision: ']))
    True_Negatives = int((test_generator.n)-True_Positives-False_Negatives-False_Positives)

    Precicion = eval_metrics['Precision: ']
    Recall = eval_metrics['Recall: ']
    Accuracy = eval_metrics['Accuracy: ']


    confusion_matrix = np.array([[True_Positives,False_Positives],[False_Negatives,True_Negatives]])

    print('Confusion Matrix:')
    print(confusion_matrix)

    print('Accuracy:')
    print(Accuracy)

    print('Precicion:')
    print(Precicion)

    print('Recall:')
    print(Recall)