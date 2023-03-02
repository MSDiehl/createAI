from attentionMechanisms import *
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense, Attention
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


#Where encoder_input_data, decoder_input_data, and decoder_target_data are your training data

#batch_size, epochs, input_vocab_size, output_vocab_size, input_length, output_length, and n_units are hyperparameters that you can adjust based on your specific task.

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(dataset.batch(batch_size),
    epochs=num_epochs,
    validation_data=(x_test, y_test))


