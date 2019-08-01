import pandas as pd
import numpy as np

import keras
from keras import models
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.layers import Activation
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.layers import BatchNormalization, Dropout, AveragePooling2D, GlobalAvgPool2D
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.applications import inception_v3, DenseNet121
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import categorical_crossentropy
from keras.layers.core import Dense, Flatten


from tensorflow import set_random_seed
set_random_seed(1112)

from set_img import *

def set_cnn_model(input_shape=(90, 120, 3)):
    '''
    Set a convolutional neural network (cnn) model 
    with one convolutional layer and one dense layer
    
    Optional Parameter
        input_shape: image pixel dimensions and channels, by default (90,120,3)
    Return
        cnn
    '''
    
    cnn = models.Sequential()
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', 
                   dilation_rate=(2, 2), kernel_regularizer=l2(0.02), input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(32, activation='relu'))
    cnn.add(Dropout(0.3))
    cnn.add(BatchNormalization())
    cnn.add(Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_model(cnn, epoch=20, batch_num=32, reduce_by=0.5, start_epoch=0):
    ''' 
    Fit the convolutional neural network (cnn) model 
    
    Parameter
        cnn: model to be fitted   
    Optional Parameters
        epoch: number of epochs, 20 by default
        batch_num: batch number, 32 by default
        reduce_by: a multiplier for to reduce steps_per_epoch, 0.5 by default
        start_epoch: start number of epoch, by default 0
    Return
        test_data
    '''
    
    # Preporcess images and load them in the ImageDataGenerator
    test_data, train_data, valid_data = split_images()
    # Use Stochastic gradient descent and set the learning rate and momentum
    optimize = SGD(lr=1e-1, momentum=0.9) 
    # Compile the model
    cnn.compile(loss='categorical_crossentropy',
                optimizer=optimize,
                metrics=['accuracy'])
    # Set reduce learning rate on plateau
    rlrop = ReduceLROnPlateau(
                                monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=1e-4, 
                                min_lr=1e-6
                            )
    # Chekpoint to save the model and its weights
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/First_CNN_111.h5', verbose=1)
    
    # Fit the model
    cnn.fit_generator(train_data,
                       steps_per_epoch= 200 * reduce_by,
                       # np.ceil(train_data.samples/train_data.batch_size),
                       epochs=start_epoch, 
                       verbose=1,
                       callbacks=[checkpoint, rlrop], 
                       validation_data=valid_data, 
                       validation_steps=32,
                       workers=16, 
                       use_multiprocessing=True, 
                       shuffle=True
                       )
    return test_data

def set_cnn_bn_l2_model(input_shape=(90, 120, 3)):
    '''
     Set a convolutional neural network (cnn) model 
    with tow convolutional layers, one dense layer, and l2 regularization
    
    Optional Parameter
        inp_shape: image pixel dimensions and channels, by default (90,120,3)
    Return
        cnn
    '''
    
    cnn = models.Sequential()
    cnn.add(Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', 
            dilation_rate=(2, 2), kernel_regularizer=l2(0.02),input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.02)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2))) 
    cnn.add(Flatten())
    cnn.add(layers.Dense(32, activation='relu', kernel_regularizer=l2(0.02)))
    cnn.add(Dropout(0.5))
    cnn.add(BatchNormalization())
    cnn.add(Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_bn_l2_model(cnn, batch_num=32, loss_param='categorical_crossentropy', epoch=20, reduce_by=0.5):
    '''
    Fit the regularized convolutional neural network (cnn) model 
    
    Parameter
        cnn: model to be fitted   
    Optional Parameters
        batch_num: batch number, 32 by default
        loss_param: loss parameter, categorical_crossentropy by default
        epoch: number of epochs, 20 by default
        reduce_by: a multiplier for to reduce steps_per_epoch, 0.5 by default
    Return
        test_data
    '''  
    
    callbacks_list = []
    
    # Preporcess images and load them in the ImageDataGenerator
    test_data, train_data, valid_data = split_images()
    # Set reduce learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5) 
    # Use Stochastic gradient descent and set the learning rate and momentum
    sgd = SGD(lr=0.1, momentum=0.95, nesterov=False)
    # Compile model
    cnn.compile(loss=loss_param,
                optimizer=sgd,
                metrics=['accuracy'])
    # Chekpoint to save the model and its weights
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/cnn_bnn_balanced_2.h5', verbose=1)
    
    # Fit model
    cnn.fit_generator(train_data,
                       steps_per_epoch= 200 * reduce_by,
#                       np.ceil(train_data.samples/train_data.batch_size),
                       epochs=epoch, 
                       verbose=1,
                       callbacks=[checkpoint], 
                       validation_data=valid_data, 
                       validation_steps=np.ceil(valid_data.samples/valid_data.batch_size), 
                       workers=16, 
                       use_multiprocessing=True, 
                       shuffle=True)
    return test_data

def dense_model(epoch, num_classes=7, inp_shape=(90, 120, 3), reduce_by=0.5):
    '''
     Build and fit a pretrained Dense121 model
    
    Parameter
        epoch: number of epochs 
    Optional Parameters
        num_classes: number of multiclass ouput, 7 by default
        inp_shape: image pixel dimensions and channels, by default (90,120,3)
        reduce_by: a multiplier for to reduce steps_per_epoch, 0.5 by default
    Return
        test_data, model 
    '''

    base_model = DenseNet121(weights=None, include_top=False, input_shape=inp_shape)
    x = AveragePooling2D(pool_size=(2,2), name='avg_pool')(base_model.output) 
    BatchNormalization()
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    BatchNormalization()
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)
    
    
    # Print layer names and its indices 
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    # Freeze layers to prevent them from getting trained
    for layer in model.layers[:427]:
        layer.trainable = False
    # Train the following layers
    for layer in model.layers[427:]:
        layer.trainable = True
    
    # Preporcess images and load them in the ImageDataGenerator
    test_data, train_data, valid_data = split_images()
    # Use Stochastic gradient descent and set the learning rate and momentum
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=False)
    # Set reduce learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10) 
    # Compile pretrained model
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit model
    model.fit_generator(generator=train_data,
                    validation_data=valid_data,
                    validation_steps = 32,
                    steps_per_epoch = 200 * reduce_by,
                    callbacks=[rlrop],
                    epochs=epoch)
    # Save model
    model.save('../SkinCancerClassifier/saved_models/densenet121_101.h5')
    
    return test_data, model

def imagenet_classifier(inp_shape=(90, 120, 3)):
    '''
    Configure a pretrained InceptionV3 image classifier 
    and add 4 dense layers with l2 regularization
    
    Optional Parameter
        inp_shape: image pixel dimensions and channels, by default (90,120,3)
    Return
        new_imagenet (the model)
    '''
    imagenet=inception_v3.InceptionV3(weights='imagenet',include_top=False, input_shape=inp_shape)
    imagenet_new =imagenet.output
    new_imagenet = models.Sequential()
    new_imagenet.add(imagenet)
    new_imagenet.add(GlobalAvgPool2D())
    new_imagenet.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(512, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(128, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(7,activation='softmax')) #final layer with softmax activation
    
    return new_imagenet
 
def train_imagenet_classifier(new_imagenet, epoch=30, reduce_by=0.5):   
    '''
    Fit pretrained InceptionV3 image classifier
    
    Parameter
        new_imagenet: model 
    Optional Parameters
        epoch: number of epochs, 30 by default
        reduce_by: a multiplier for to reduce steps_per_epoch, 0.5 by default
    Return
        test_data, new_imagenet
    '''
        
    # Print all layers in imagenet
    for i, layer in enumerate(new_imagenet.layers):
        print(i, layer.name)
    
    # Flag layers to be frozen so they don't get trained
    for layer in new_imagenet.layers[:1]:
        layer.trainable=False
    
    # Set learning rate plateau
    rlrop = ReduceLROnPlateau(
                                monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=1e-4, 
                                min_lr=1e-6
                            )
    # Use Stochastic Gradient Descent with set learnig rate and momentum
    sgd = SGD(lr=0.1, momentum=0.9)
    # Set checkpoint to save model and weights
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/imagenet_1157.h5', 
                                  verbose=1)
    # Compile model
    new_imagenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_imagenet.summary())
    # Preporcess images and load them in the ImageDataGenerator
    test_data, train_data, valid_data = split_images()
    # Fit model
    new_imagenet.fit_generator(generator=train_data,
                    validation_data=valid_data,
                    validation_steps = 32,
                    steps_per_epoch = 200 * reduce_by, 
                    callbacks=[rlrop, checkpoint],
                    epochs=epoch)

    return test_data, new_imagenet

def set_cnn_no_dense_model(input_shape = (90, 120, 3), num_classes = 7):
    '''
    Configure a convolutional neural network with 11 convolutions
    
    Optional Parameter
        inp_shape: image pixel dimensions and channels, by default (90,120,3)
        num_classes: number of multiclass ouput, 7 by default
    Return
        model
    '''
    
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3, 3),activation='relu',name="Conv1", input_shape=input_shape)) 
    model.add(BatchNormalization(name="Norm1"))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',name="Conv2")) 
    model.add(BatchNormalization(name="Norm2"))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',name="Conv3")) 
    model.add(BatchNormalization(name="Norm3"))
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Dropout(0.20))

    model.add(Conv2D(64, (3, 3), activation='relu',name="Conv4")) 
    model.add(BatchNormalization(name="Norm4"))
    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv5")) 
    model.add(BatchNormalization(name="Norm5"))
    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv6")) 
    model.add(BatchNormalization(name="Norm6"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv7")) 
    model.add(BatchNormalization(name="Norm7"))
    model.add(Conv2D(256, (3, 3), activation='relu',name="Conv8")) 
    model.add(BatchNormalization(name="Norm8"))
    model.add(Conv2D(256, (3, 3), activation='relu',name="Conv9")) 
    model.add(BatchNormalization(name="Norm9"))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(0.20))

    model.add(Conv2D(7,(1,1),name="conv10",activation="relu")) 
    model.add(BatchNormalization(name="Norm10"))
    model.add(Conv2D(7,kernel_size=(6,9),name="Conv11"))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    print(model.summary())
    
    return model

def fit_cnn_no_dense_model(model, start_at_epoch,  epoch, reduce_by=0.5):   
    '''
    Fit 11 layer deep cnn model 
    
    Parameter
        model: model to be fitted  
        start_at_epoch: epoch number as a starting point
        epoch: number of epochs 
    Optional Parameters
        reduce_by: a multiplier for to reduce steps_per_epoch, 0.5 by default
    Return
        test_data, model
    '''
    
    # Compile the model
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])
    # Set learning rate plateau 
    rlrop = ReduceLROnPlateau(monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=1e-4, 
                                min_lr=1e-6)
    # Checkpoint to save model and weights
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/cnn_nodense_model3.h5', 
                                   verbose=1)
    # Preporcess images and load them in the ImageDataGenerator
    test_data, train_data, valid_data = split_images()
    
    # Fit model
    model.fit_generator(generator=train_data,
                        validation_data=valid_data,
                        validation_steps = 32,
                        steps_per_epoch = 200 * reduce_by,
                        callbacks=[rlrop, checkpoint],
                        initial_epoch=start_at_epoch,
                        epochs=epoch)
    
    return test_data, model

def load_display_model(file_path, target_size=(90,120)):
    '''
    Load saved model/weights, evaluate, predict on test data, 
    plot confusion matrix and display classification report
    
    Parameter
        file_path: file path of saved model/weights
    Optional Parameter
        target_size: image size in pixels, 90 x 120 by default
    '''
    test_data, _, _ = split_images(target_size=target_size)
    te_img, te_label = next(test_data)
    model = load_model(file_path)
    model.evaluate(te_img, te_label)
    y_pred = model.predict(te_img)
    plot_confusion_matrix(te_label.argmax(axis=1), y_pred.argmax(axis=1))
    