"""
TrainModel:
    1. Imports all preprocessing data and variables
    2. Sets a desired accuracy for the model
    3. Builds a learning model 
    4. compiles the model using custom optimizer and loss function
    5. prints model summary
    6. defines a callback class and initializes a callback for each epoch end
    7. Sets the number of epoch to train the model
    8. fits/trains the model on preprocessed training dataset (Predictors and Labels)
    9. saves the trained model as "TrainedModel.h5"
    10. extracts the comaprisom metrics from the trained model (accuracy and loss)
    11. plots the extracted comaprison metrics and save sthe images to "Images/"
"""
import BuildModel as BM
import DataPreprocessing as DP
import PlotCode as PC
import tensorflow as tf


TOTAL_WORDS = DP.total_words
MAX_SEQ_LEN = DP.max_sequence_length
PREDICTORS = DP.predictors
LABEL = DP.label
DESIRED_ACC = 0.95

EMBEDDING_DIM = 256
Model = BM.BuildModel()
model = Model.build_model(TOTAL_WORDS, EMBEDDING_DIM, 256, 128, TOTAL_WORDS, TOTAL_WORDS, MAX_SEQ_LEN - 1)
# model = BM.build_model(TOTAL_WORDS, EMBEDDING_DIM, 256, 128, TOTAL_WORDS, TOTAL_WORDS, MAX_SEQ_LEN - 1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > DESIRED_ACC):
            print("\n reached desired accuracy, so stoping further training")
            self.model.stop_training = True

callbacks = myCallback()


EPOCHS = 100
history = model.fit(PREDICTORS, LABEL, epochs = EPOCHS, verbose = 1, callbacks = [callbacks])


model.save("TrainedModel.h5")

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(acc))

PC.plot(acc, epochs, 'accuracy', 'b')
PC.plot(loss, epochs, 'loss', 'r')