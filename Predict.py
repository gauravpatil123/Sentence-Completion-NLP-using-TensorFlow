import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import DataPreprocessing as DP

tokenizer = DP.tokenizer
max_seq_len = DP.max_sequence_length

model = tf.keras.models.load_model("TrainedModel.h5")
SEED_TEXT = "To quit"
NUM_NEXT_WORDS = 5

def predict(seed_text, num_next_words):
    for _ in range(num_next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_seq_len - 1, padding = 'pre')
        predicted = model.predict_classes(token_list, verbose = 0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

PREDICT = predict(SEED_TEXT, NUM_NEXT_WORDS)
print(PREDICT)

