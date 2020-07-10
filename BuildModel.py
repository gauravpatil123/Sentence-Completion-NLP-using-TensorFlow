import tensorflow as tf
from tensorflow.keras.layers import LSTM

def build_model(vocab_size, embedding_dim, lstm_layer1_units, lstm_layer2_units, dense_units, output_units, inputlen):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=inputlen),
            #tf.keras.layers.Conv1D(64, 5, activation = 'relu'),
            #tf.keras.layers.MaxPooling1D(pool_size = 4),
            tf.keras.layers.Bidirectional(LSTM(lstm_layer1_units, return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(LSTM(lstm_layer2_units)),
            tf.keras.layers.Dense(dense_units, activation = 'relu'),
            tf.keras.layers.Dense(output_units, activation = 'softmax')
    ])
    return model