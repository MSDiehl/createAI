import tensorflow as tf
<<<<<<< HEAD
from attentionMechanisms import *
=======
>>>>>>> e01ac3fba0b2505e83e37913163f940880a0e460
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense, Attention

model = create_attention_model(input_vocab_size, output_vocab_size, input_length, output_length, n_units)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
<<<<<<< HEAD
          validation_split=0.2)
=======
          validation_split=0.2)
>>>>>>> e01ac3fba0b2505e83e37913163f940880a0e460
