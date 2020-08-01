"""
DataPreprocessing:
    1. opens, reads and preprocesses training dataset into corpus
    2. tokenizes the corpus
    3. creates input sequences based on n_gram sequence model
    4. pads the input sequences nased on the max length input sequences
    5. initializes predictors and labels on the dataset
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import numpy as np

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