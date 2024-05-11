# Prediction of smartphone price based on its attributes using ML/neural network (ANN and CNN)
# Normalize and categorize the variables

import numpy as np
import pandas as pd

# Read the data
df = pd.read_csv('C:\\Users\myamy\OneDrive\Desktop\Data\mobile.csv')

x = df.drop("price_range", axis=1) #  x= explanatory variable
y= df["price_range"] # y= target variable or label

# Rename columns with feature names
feature_names = ['battery_power','bluetooth','clock_speed','dual_sim', 
                 'frontcamerapixel', '4G', 'memorysize', 'weight', 
                 'cores', 'primarycamerapixel', 'pixelresolutionheight', 
                 'pixelresolutionwidth', 'ram', 'screenheight', 'screenwidth', 
                 'talk_time', '3G', 'touch_screen', 'wifi']
x.columns = feature_names

# Data partition to 70% training data and 30% to testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.3) # 30% to testing data set

# Check the split
X_train.shape
X_test.shape

# Normalize the data to remove skeweness
from sklearn.preprocessing import StandardScaler
sc  = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create artificial neural network (ANN)

# Create model in 2 steps
# Step 1: Configure the model
import tensorflow as tf
import tensorflow as keras
from tensorflow.keras import layers

# Create a sequential model
model_1 = tf.keras.Sequential()  # Add tf with keras.Sequential
# Add the layers to the model
model_1.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],))) # input layer
model_1.add(tf.keras.layers.Dense(32, activation="relu")) # second layer # random neuron
model_1.add(tf.keras.layers.Dense(4, activation="softmax"))

# Step 2:Compile the model with optimizer, loss function, and metrics
model_1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 
# Train the model on training data using artificial neural network
history = model_1.fit(X_train,y_train,epochs=30)
model_1.summary()

# Evaluate the model on the test data using artificial neural network
loss, accuracy = model_1.evaluate(X_test, y_test)
print("loss:", loss)
print("Accuracy:", accuracy)

# Create convolutional neural network (CNN)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model: convolutional neural network

# Create a sequential model.
model_2 = keras.Sequential()

# Add a convolutional layer to the model. In this layer, we define the number of filters, kernel size, activation function, and input shape.
model_2.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(19,1)))

# Add a max pooling layer to the model. This layer reduces the dimensionality of the data.
model_2.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

# Flatten the data to prepare it for the dense layers.
model_2.add(keras.layers.Flatten())

# Add one or more dense layers to the model. The number of neurons in the dense layers can be adjusted based on the complexity of the data.
model_2.add(keras.layers.Dense(64, activation='relu'))
model_2.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model with a loss function, optimizer, and metrics.
model_2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on train dataset
history = model_2.fit(X_train, y_train, epochs=30)
# 
model_2.summary()

# Evaluate the model on the test data
loss, accuracy = model_2.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Predict smart phone’s price range based on new data.

new_data_df = pd.DataFrame({'battery_power': [1500],'bluetooth':[1],'clock_speed':[2.1],'dual_sim':[1], 
                 'frontcamerapixel':[10], '4G':[1], 'memorysize':[45], 'weight':[148], 
                 'cores':[8], 'primarycamerapixel':[7], 'pixelresolutionheight':[1216], 
                 'pixelresolutionwidth':[1786], 'ram':[3763], 'screenheight':[14], 'screenwidth':[9], 
                 'talk_time':[13], '3G':[1], 'touch_screen':[1], 'wifi':[1]})

# apply scaler on the new dataframe
new_data_scaled = sc.transform(new_data_df)

# make prediction using the trained model_1 (ANN model)
prediction = model_1.predict(new_data_scaled)

# get the predicted class label
predicted_class = np.argmax(prediction[0])

# print the predicted price range
class_labels = ["low", "middle", "expensive", "very expensive"]
print("Model_1 (ANN Model):",class_labels[predicted_class])

# Predict smart phone’s price range based on new data. Based on model_2

# make prediction using the trained model_2 (CNN Model)
prediction = model_2.predict(new_data_scaled)

# get the predicted class label
predicted_class = np.argmax(prediction[0])

# print the predicted price range
class_labels = ["low", "middle", "expensive", "very expensive"]
print("Model_2 (CNN Model):",class_labels[predicted_class])

# Conclusion: We found the accuracy level 1 in ANN model based on training data. And .908 accuracy level when we evaluate the results by testing on the test dataset.
# We found the accuracy level .9121 in CNN model based on training data. And .8117 accuracy level when we evaluate the results by testing on the test dataset.
# In predicting smart phone’s price range based on new data we have found that the price of the smart phone is very expensive using artificial neural network model (ANN) and convolutional neural network model (CNN).
# In this neural networks examination ANN performed better than CNN model.