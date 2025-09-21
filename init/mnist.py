import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

reference_model_path = Path('reference_models')
dataset_model_path = Path('datasets/mnist')

reference_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
dataset_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Validation split from training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Normalize input data
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)


# Export training, validation, and test data
np.save(dataset_model_path.joinpath('x_train.npy'), x_train)
np.save(dataset_model_path.joinpath('y_train.npy'), y_train)
np.save(dataset_model_path.joinpath('x_val.npy'), x_val)
np.save(dataset_model_path.joinpath('y_val.npy'), y_val)
np.save(dataset_model_path.joinpath('x_test.npy'), x_test)
np.save(dataset_model_path.joinpath('y_test.npy'), y_test)

# Define model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='elu'),
    Dropout(0.2),
    Dense(128, activation='elu'),
    Dropout(0.2),
    Dense(128, activation='elu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model with EarlyStopping on validation data
_ = model.fit(x_train, y_train,
          epochs=50,
          batch_size=64,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
          validation_data=(x_val, y_val))

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}')


# Export model
model.save(reference_model_path.joinpath('mnist_base.keras'))
