import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

data = diabetes_dataset['data']
targets = diabetes_dataset['target']
y_binary = (diabetes_dataset.target > np.mean(diabetes_dataset.target)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(data, y_binary, test_size=.1)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
  tf.keras.layers.Dense(15, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Wyznaczenie metryk dla zbioru walidacyjnego
test_loss, test_accuracy, test_mae = model.evaluate(X_test, y_test)
print("Wartość funkcji straty dla zbioru testowego: "+str(test_loss))
print("Wartość dokladnosci dla zbioru testowego: "+str(test_accuracy))
print("Wartość funkcji MSE dla zbioru testowego: "+str(test_mae))



biggerModel = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
  tf.keras.layers.Dense(15, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(8, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

biggerModel.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mae', 'accuracy'])

biggerHistory = biggerModel.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

test_loss_bigger, test_accuracy_bigger, test_mae_bigger = biggerModel.evaluate(X_test, y_test)
print("Wartość funkcji straty dla zbioru testowego: "+str(test_loss_bigger))
print("Wartość dokladnosci dla zbioru testowego: "+str(test_accuracy_bigger))
print("Wartość funkcji MSE dla zbioru testowego: "+str(test_mae_bigger))
