'''
Project 4 : Behavioral Cloning
Script for creating and training a model
'''

import argparse
import os
import numpy as np
import pandas as pd
import random
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Lambda, Conv2D, MaxPooling2D, ELU, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

STEERING_OFFSET = 0.25

# load data from training data file
def load_data(csv_path, side_camera):
    # first row will be taken as header row
    data = pd.read_csv(csv_path)
    data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data['steering'] = data['steering'].astype(np.float32)
    
    # we need only streeing angle
    data.drop(['throttle', 'brake', 'speed'], axis=1, inplace=True)
    
    new_columns = ['image', 'steering']
    
    # use images from left and right cameras    
    if side_camera:
        # center camera
        center_data = data[['center', 'steering']]
        center_data.columns = new_columns
        
        # adjust steering angle for left and right camera images        
        # left camera
        left_data = data[['left', 'steering']]
        left_data.columns = new_columns
        # left_data = left_data[left_data['steering'] > 0.0]
        left_data['steering'] = left_data['steering'] + STEERING_OFFSET
        
        # right camera
        right_data = data[['right', 'steering']]
        right_data.columns = new_columns
        # right_data = right_data[right_data['steering'] < 0.0]
        right_data['steering'] = right_data['steering'] - STEERING_OFFSET
        
        # combine
        data = pd.concat([center_data, left_data, right_data], axis=0, ignore_index=True)
        
    else:
        # use only images from center camera
        data.drop(['left', 'right'], axis=1, inplace=True)
    
    # split data into features and labels
    X = data.iloc[:,0].values
    y = data.iloc[:,1].values.astype(np.float32)
    
    return X, y


# create a model based on reference paper. See report.
def create_model():
    model = Sequential()
    # Normalization (-1 to 1)
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", kernel_initializer='he_normal'))    
    # model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(ELU())
              
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", kernel_initializer='he_normal')) 
    model.add(BatchNormalization())
    model.add(ELU())
    
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", kernel_initializer='he_normal')) 
    model.add(BatchNormalization())
    model.add(ELU())
   
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal')) 
    model.add(BatchNormalization())
    model.add(ELU())
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal')) 
    model.add(BatchNormalization())
    model.add(ELU()) 
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(ELU())
    
    model.add(Dense(100, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(50, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

def load_image(img_path):
    nvidia_w, nvidia_h = 200, 66
    image = cv2.imread(img_path) 
    
    # randomly adjust brightness
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright_factor = 0.25 * np.random.uniform()
    image[:, :, 2] = image[:, :,2]*bright_factor    
   
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    # crop the sky at the top of the image
    image = image[50:-20]
    
    image = cv2.resize(image, (nvidia_w, nvidia_h), interpolation=cv2.INTER_AREA)
    
    return image

# custom data generator with data augmentation
def image_data_generator(data_dir, X, y, batch_size=32):
    num_samples = len(X)
    batch_until_shuffle = num_samples//batch_size
    
    batch_counter, start_idx = 0, 0
    center_image_limit = batch_size//2
    center_image_counter = 0
    
    features = np.ndarray(shape=(batch_size, 66, 200, 3), dtype=np.float32)
    labels = np.ndarray(shape=(batch_size,), dtype=np.float32)
    
    while True:
        # shuffle after a complete parse of dataset
        if (batch_counter % batch_until_shuffle) == 0:
            idx = np.arange(0, num_samples)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            start_idx = 0
        for i in range(start_idx, start_idx + batch_size):
            img_path, label = X[i], y[i]
            
            # limit the number of 'steering => 0' images in each batch 
            # i.e boost the number of side_camera images
            if abs(label) < 0.1:
                center_image_counter += 1
            if center_image_counter > center_image_limit:
                random_idx = random.choice(range(0, num_samples))
                while abs(y[random_idx] < 0.1):
                    random_idx = random.choice(range(0, num_samples))
                img_path, label = X[random_idx], y[random_idx]
                
            img_path = img_path.strip()
            img_path = data_dir + '/' + img_path
            image = load_image(img_path)
            image = np.array(image, dtype=np.float32)
    
            # randomly flip the image
            if random.choice([0,1]) == 1:
                image = cv2.flip(image, 1)
                label = -label
            
            features[i % batch_size] = image
            labels[i % batch_size] = label
            
        batch_counter += 1
        start_idx += batch_size        
        yield (features, labels)
        
if __name__=='__main__':    
    # Argument Parser
    parser = argparse.ArgumentParser(description='Behavioral Cloning : Steering Angle Prediction')
    parser.add_argument('--data_dir', action = 'store', dest='data_dir', type=str, default='../../../opt/carnd_p3/data', help='Provide path to training data')
    parser.add_argument('--epochs', action = 'store', dest='epochs', type=int, default=5, help='Number of Epochs')
    parser.add_argument('--batch_size', action = 'store', dest='batch_size', type=int, default=32, help='Training Batch Size')
    parser.add_argument('--side_camera', action = 'store_true', dest='side_camera', help='Train using left and right camera images too')
    parser.add_argument('--continue_training', action = 'store_true', dest='continue_training', help='Continue training from the previous model')
    parser.add_argument('--model_file', action = 'store', dest='model_file', type=str, default='model.h5', help='Model destination')
    
    # Parse arguments
    args = parser.parse_args()
    
    data_dir = str(args.data_dir)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    side_camera = args.side_camera
    continue_training = args.continue_training
    model_file = str(args.model_file)
    
    
    print('------- Selected Options -------')
    print('data_dir       = {}'.format(data_dir))
    print('epochs           = {}'.format(epochs))
    print('batch_size       = {}'.format(batch_size))
    print('side_camera      = {}'.format(side_camera))
    print('continue_training= {}'.format(continue_training))
    print('model_file       = {}'.format(model_file))
    print('--------------------------------')
    
    # load training data
    csv_path = data_dir + '/driving_log.csv'
    
    print('Loading data from file: {}'.format(csv_path))
    X, y = load_data(csv_path, side_camera)
    
    # split data into training and validation sets
    # using most of the data for training. See report.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    train_size = y_train.shape[0]
    valid_size = y_valid.shape[0]
    print('Size of Training dataset: {}'.format(train_size))
    print('Size of Validation dataset: {}'.format(valid_size))
    
    # Modle training
    if continue_training:
        print('Loading model from {} file'.format(model_file))
        model = load_model(model_file)
    else:
        model = create_model()
        
    print(model.summary())
    
    steps_per_epoch = train_size//batch_size
    validation_steps = valid_size//batch_size
    
    history = model.fit_generator(image_data_generator(data_dir, X_train, y_train, batch_size), steps_per_epoch=steps_per_epoch,
                        epochs=epochs, validation_data=image_data_generator(data_dir, X_valid, y_valid, batch_size),
                        validation_steps=validation_steps, verbose=1)
   
    
    # update model file
    if os.path.isfile(model_file):
        os.remove(model_file)
    model.save(model_file)
    
    print('Training Successful. Model saved in {} '.format(model_file))
    print('---------- Done ---------------')