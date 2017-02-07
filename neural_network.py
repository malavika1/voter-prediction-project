import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
import utilities as ut

# Get the normalized data
X_train, y_train, X_test = ut.import_data(one_d_array=False)
y_train -= 1

# Create a model
model = Sequential()
for i in range(10):
    model.add(Dense(50, input_shape=(X_train.shape[1],), init='uniform', activation='relu'))
    model.add(Dropout(0.4))

# Predict the probabilities
model.add(Dense(1, activation='sigmoid'))

# Printing a summary of the layers and weights
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=1)

print(model.evaluate(X_train, y_train))

# Printing the test data based on the model
predictions = [i[0]+1 for i in model.predict_classes(X_test)]
ut.write_output_file(np.array(predictions), file_name='neural_net.csv')
