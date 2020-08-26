"""
DataPreprocessing:
    1. Defines a class ProcessData to process teh dataset to be trained
    2. Initializes a object processed_data to be used dynamically in
       Training and prediction scripts
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import numpy as np


class ProcessData:

    """
    Class to preprocess the dataset to be trained
    """

    def __init__(self, data_file_dir):
        """
        Input:
            data_file_dir: file directory location (as string)

        Actions:
            1. Initializes the Tokenizer
            2. Initializes and reads the data from the data_file_dir directory location
            3. Initialized corpus
            4. Fits the corpus on the Tokenizer
            5. Initilazes total words
            6. Initializes the input sequences using the corpus and Tokenizer
            7. Initializes the max sequence length from the input sequences
            8. Initilaizes the predictors and labels from the input sequences
        """
        self.tokenizer = Tokenizer()
        self.data = open(data_file_dir).read()
        self.corpus = data.lower().split("\n")
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1
        
        self.input_sequences = []
        for line in self.corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                self.input_sequences.append(n_gram_sequence)

        self.max_sequence_length = max([len(x) for x in self.input_sequences])
        self.input_sequences = np.array(pad_sequences(self.input_sequences, maxlen=self.max_sequence_length, padding='pre'))

        self.predictors, self.label = self.input_sequences[:,:-1], self.input_sequences[:,-1]
        self.label = ku.to_categorical(self.label, num_classes=self.total_words)

    def __call__(self):
        """
        Returns the total words, max sequence length, prdictors and labels that are already initialized
        """
        return self.total_words, self.max_sequence_length, self.predictors, self.label

    def get_tokenizer(self):
        """
        Returns the initialized Tokenizer
        """
        return self.tokenizer

    def get_total_words(self):
        """
        Returns the initialized total words
        """
        return self.total_words

    def get_predictors(self):
        """
        Returns the initialized predictors
        """
        return self.predictors

    def get_label(self):
        """
        Returns the initialized labels
        """
        return self.label

    def get_max_seq_len(self):
        """
        Returns the initialized max sequence length
        """
        return self.max_sequence_length

# Initializing dynamic object processed_data of the class ProcessData 
# to be used in TrainModel and Predict
processed_data = ProcessData("data/data.txt")

"""
tokenizer = Tokenizer()

data = open('data/data.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#print(input_sequences[:150])

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_length, padding = 'pre'))

predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = ku.to_categorical(label, num_classes = total_words)
#print(predictors[:50, :])
#print(max_sequence_length)
"""
