"""
TrainModel:
    1. Imports the processed_data object of the class ProcesData
    2. Calls the processed_data object to initialize the processed data variables and dataset 
    3. Sets a desired accuracy for the model
    4. Builds a learning model 
    5. compiles the model using custom optimizer and loss function
    6. prints model summary
    7. defines a callback class and initializes a callback for each epoch end
    8. Sets the number of epoch to train the model
    9. fits/trains the model on preprocessed training dataset (Predictors and Labels)
    10. saves the trained model as "TrainedModel.h5"
    11. extracts the comaprisom metrics from the trained model (accuracy and loss)
    12. plots the extracted comaprison metrics and saves the images to "Images/"
"""
import BuildModel as BM
import DataPreprocessing as DP
import PlotCode as PC
import tensorflow as tf

PROCESSED_DATA = DP.processed_data
TOTAL_WORDS, MAX_SEQ_LEN, PREDICTORS, LABEL = PROCESSED_DATA()

DESIRED_ACC = 0.95

EMBEDDING_DIM = 256
Model = BM.BuildModel()
model = Model.build_model(TOTAL_WORDS, EMBEDDING_DIM, 256, 128, TOTAL_WORDS, TOTAL_WORDS, MAX_SEQ_LEN - 1)

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

acc_graph = PC.Plot(acc, epochs, 'accuracy', 'b')
acc_graph()

loss_graph = PC.Plot(loss, epochs, 'loss', 'r')
loss_graph()
