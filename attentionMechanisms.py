import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense, Attention

def create_attention_model(input_vocab_size, output_vocab_size, input_length, output_length, n_units):
    # Encoder
    inputs = Input(shape=(input_length,))
    encoder_emb = Embedding(input_dim=input_vocab_size, output_dim=n_units)(inputs)
    encoder_lstm = Bidirectional(LSTM(n_units, return_sequences=True, dropout=0.5))(encoder_emb)
    encoder_lstm_2 = Bidirectional(LSTM(n_units, return_sequences=True, dropout=0.5))(encoder_lstm)

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(output_vocab_size, n_units)(decoder_inputs)
    decoder_lstm = LSTM(n_units * 2, return_sequences=True, dropout=0.5)(decoder_emb)

    # Attention
    attention = Attention()([decoder_lstm, encoder_lstm_2])

    # Concatenate
    concat = Concatenate(axis=-1)([decoder_lstm, attention])

    # Output
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    outputs = decoder_dense(concat)

    # Model
    model = tf.keras.Model([inputs, decoder_inputs], outputs)
    return model