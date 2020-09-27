import tensorflow as tf
from tensorflow.keras.layers import LSTM


class BuildModel:
        """
        Class to define the neural network model
        """

        def __init__(self):
                """
                Initializes model to None
                """
                self.model = None

        def build_model(self, vocab_size, embedding_dim, lstm_layer1_units, 
                        lstm_layer2_units, dense_units, output_units, inputlen):
                """
                Input:
                    vocab_size: length of the vocabulary size (total words)
                    embedding_dim: number of embedding dimensions
                    lstm_layer1_units: number of units in the first LSTM layer
                    lstm_layer2_units: number of units in the second LSTM layer
                    dense_units: number of neurons in the fully connected hidden layer
                    output_units: number of neurons in the output layer
                    inputlen: max length of input data
                Output:
                    model: the neural network model built using the inputs
                """
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=inputlen),
                    tf.keras.layers.Bidirectional(LSTM(lstm_layer1_units, return_sequences=True)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Bidirectional(LSTM(lstm_layer2_units)),
                    tf.keras.layers.Dense(dense_units, activation = 'relu'),
                    tf.keras.layers.Dense(output_units, activation = 'softmax')
                ])
                return self.model

        def load_model(self, model_name):
                """
                Input:
                    model_name: path/name of the model file to be loaded
                Output:
                    model: the neural neteork model loaded from the file
                """
                self.model = tf.keras.models.load_model(model_name)
                return self.model