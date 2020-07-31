# Sentence-Completion-NLP-using-TensorFlow
Sentence completion model on trained on Shakespeare's "The comedy of error's" using NLP in TensorFlow and Keras

**Dataset**
- [The comedy of error's](http://shakespeare.mit.edu/comedy_errors/full.html)

**Processing Scripts**
- DataPreprocessing : Preprocessing raw data, tokenizing, forming input sequences and labels for the dataset
- PlotCode: Plotting metrics from the trained model

**Classes**
- BuildModel : model class for the neural network NLP model

**Executables**
- TrainModel : Trains the model on the training dataset
    - Model parameters used in training
        1. embedding dimensions = 256
        2. first LSTM layer units = 256
        3. second LSTM layer units = 128
        4. fully connected hidden layer neurons = TOTAL_WORDS

- Predict : Predicts the next words of the sentence on the seed text using the pretrained model

**Results**
- Accuracy and loss of the model on training dataset
- <img src="Images/accuracy.png" width=1000>
- <img src="Images/loss.png" width=1000>