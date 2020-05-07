import pandas as pd
import numpy as np
import copy

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input, Model

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from callbacks import CallbackPlot, CallbackSaveLogs

import tensorflow as tf

from data_processor import clean_data, build_datasets
from data_processor import one_hot2char, process_input

#%matplotlib
import matplotlib
import matplotlib.pyplot as plt

import sys
import json
import os
from shutil import copyfile
from datetime import datetime

MAX_LEN_TITLE = 82
MAX_LEN_TEXT = 390

# enable memory growth to be able to work with GPU
GPU = tf.config.experimental.get_visible_devices('GPU')[0]
tf.config.experimental.set_memory_growth(GPU, enable=True)

# set tensorflow to work with float64
tf.keras.backend.set_floatx('float64')

# the new line character (\n) is the 'end of sentence', therefore there is no need to add a '[STOP]' character
vocab = '\'kespw/br461 \ncunt-".x:j0ml,(qzgif)25a7o9vdyh83'
vocab = list(vocab) + ['[START]']

config = {  # dictionary that contains the training set up. Will be saved as a JSON file
    'DIM_VOCAB': len(vocab),
    'DIM_LSTM_LAYER': 1024,
    'LEARNING_RATE': 0.002,
    'BATCH_SIZE': 8,
    'EPOCHS': 40,
    'SEED': 1,
    'GRAD_VAL_CLIP': 0.5,
    'GRAD_NORM_CLIP': 1,
    'DECAY_AT_10_EPOCHS': 1/2,
    'DROPOUT': 0.1,
}
tf.random.set_seed(config['SEED'])
data = pd.read_csv('Complete Database.csv', index_col=0)

data = clean_data(data)

config['STEPS_PER_EPOCH'] = int((data.shape[0] - 3000) / config['BATCH_SIZE'])
config['VALIDATION_SAMPLES'] = int(data.shape[0]) - (config['STEPS_PER_EPOCH'] * config['BATCH_SIZE'])
config['VALIDATION_STEPS'] = int(np.floor(config['VALIDATION_SAMPLES'] / config['BATCH_SIZE']))

# configures the learning rate to be decayed by the value specified at config['DECAY_AT_10_EPOCHS'] at each 10 epochs, but to that gradally at each epoch
learning_rate = ExponentialDecay(initial_learning_rate=config['LEARNING_RATE'], 
                                decay_steps=config['STEPS_PER_EPOCH'], 
                                decay_rate=np.power(config['DECAY_AT_10_EPOCHS'], 1/10), 
                                staircase=True)

""" config['STEPS_PER_EPOCH'] = 5
config['VALIDATION_SAMPLES'] = 100
config['VALIDATION_STEPS'] = 5
 """
training_dataset, validation_dataset = build_datasets(data, seed=config['SEED'], validation_samples=config['VALIDATION_SAMPLES'], batch=config['BATCH_SIZE'], vocab=vocab)

#learning_rate = config['LEARNING_RATE']

if 'GRAD_VAL_CLIP' in config:
    optimizer = Adam(learning_rate=learning_rate, clipvalue=config['GRAD_VAL_CLIP'], clipnorm=config['GRAD_NORM_CLIP'])
    print('Using gradient clipping')
else:
    optimizer = Adam(learning_rate=learning_rate)

config['OPTIMIZER'] = optimizer._name

folder_path = 'Training Logs/Training'  # creates folder to save traning logs
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

with open(folder_path + '/config.json', 'w') as json_file: # saves the training configuration as a JSON file
    json.dump(config, json_file, indent=4)

with open(folder_path + '/vocab.json', 'w')  as json_file: # saves the vocab used as a JSON file
    json.dump(vocab, json_file, indent=4)


validation_steps = int(config['VALIDATION_SAMPLES'] / config['BATCH_SIZE'])

loss_plot_settings = {'variables': {'loss': 'Training loss',
                                        'val_loss': 'Validation loss'},
                          'title': 'Losses',
                          'ylabel': 'Epoch Loss'}

last_5_plot_settings = {'variables': {'loss': 'Training loss',
                                        'val_loss': 'Validation loss'},
                          'title': 'Losses',
                          'ylabel': 'Epoch Loss',
                          'last_epochs': 5}

plot_callback = CallbackPlot(folder_path=folder_path,
                            plots_settings=[loss_plot_settings, last_5_plot_settings],
                            title='Losses', share_x=False)

model_checkpoint_callback = ModelCheckpoint(filepath=folder_path + '/trained_model.h5')

csv_logger = CSVLogger(filename=folder_path + '/Training logs.csv', separator=',', append=False)

###### BUILDS MODEL FROM SCRATCH ###########
tf.keras.backend.clear_session() # destroys the current graph

encoder_inputs = Input(shape=(None, config['DIM_VOCAB']), name='encoder_input')
encoder_LSTM = LSTM(units = config['DIM_LSTM_LAYER'], return_sequences=False, return_state=True, name='encoder_LSTM', dropout=config['DROPOUT'])
enc_output, enc_memory_state, enc_carry_state = encoder_LSTM(encoder_inputs)

decoder_inputs = Input(shape=(None, config['DIM_VOCAB']), name='decoder_input')
decoder_LSTM = LSTM(units=config['DIM_LSTM_LAYER'], return_sequences=True, return_state=True, name='decoder_LSTM', dropout=config['DROPOUT'])  # return_state must be set in order to retrieve the internal states in inference model later
decoder_output, _, _ = decoder_LSTM(decoder_inputs, initial_state=[enc_memory_state, enc_carry_state])

dense = Dense(units=config['DIM_VOCAB'], activation='softmax', name='output')
dense_output = dense(decoder_output)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_output)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')

history = model.fit(x=training_dataset,
                    epochs=config['EPOCHS'],
                    steps_per_epoch=config['STEPS_PER_EPOCH'],
                    callbacks=[plot_callback, csv_logger, model_checkpoint_callback],
                    validation_data=validation_dataset,
                    validation_steps=config['VALIDATION_STEPS'])
                   

'''
model.save(folder_path + '/trained_model.h5', save_format='h5')
model.save_weights(folder_path + '/trained_model_weights.h5')
plot_model(model, to_file=folder_path + '/model_layout.png', show_shapes=True, show_layer_names=True, rankdir='LR')
'''

timestamp_end = datetime.now().strftime('%d-%b-%y -- %H:%M:%S')

# renames the training folder with the end-of-training timestamp
root, _ = os.path.split(folder_path)

timestamp_end = timestamp_end.replace(':', '-')
os.rename(folder_path, root + '/' + 'Training Session - ' + timestamp_end)

print("Training Succesfully finished.")