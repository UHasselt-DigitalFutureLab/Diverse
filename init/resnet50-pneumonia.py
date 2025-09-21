import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from medmnist import PneumoniaMNIST
from tensorflow.keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

reference_model_path = Path('reference_models')
dataset_model_path = Path('datasets/pneumonia_mnist')

reference_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
dataset_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


# Helper to convert to numpy
def extract_data(dataset):
    images = dataset.imgs  # shape: (N, 28, 28, 3)
    labels = dataset.labels  # shape: (N, 1)
    return images, labels


def to_rgb(x):
    # x: (N, 224, 224) or (N, 224, 224, 1)  →  (N, 224, 224, 3)
    if x.ndim == 3:
        x = x[..., np.newaxis]          # (N,H,W,1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)    # (N,H,W,3)
    return x.astype("float32")


# Load dataset from medmnist
train_dataset = PneumoniaMNIST(split='train', size=224, download=True)
val_dataset = PneumoniaMNIST(split='val', size=224, download=True)
test_dataset = PneumoniaMNIST(split='test', size=224, download=True)


X_train, y_train = extract_data(train_dataset)
X_val, y_val = extract_data(val_dataset)
X_test, y_test = extract_data(test_dataset)


X_test = to_rgb(X_test)
X_train = to_rgb(X_train)
X_val = to_rgb(X_val)

X_test = preprocess_input(X_test)
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)


# Convert y values from [0, 1] to one-hot encoded shape (n_samples, 2)
y_train = to_categorical(y_train, num_classes=2)
y_val   = to_categorical(y_val, num_classes=2)
y_test  = to_categorical(y_test, num_classes=2)


# Save normalized, one-hot encoded data
np.save(dataset_model_path.joinpath("x_train_normalized.npy"), X_train)
np.save(dataset_model_path.joinpath("x_test_normalized.npy"), X_test)
np.save(dataset_model_path.joinpath("x_val_normalized.npy"), X_val)
np.save(dataset_model_path.joinpath("y_train_onehot.npy"), y_train)
np.save(dataset_model_path.joinpath("y_test_onehot.npy"), y_test)
np.save(dataset_model_path.joinpath("y_val_onehot.npy"), y_val)



model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False  # Freeze the base model
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Optional
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

early_stop = EarlyStopping(
    monitor='val_loss',    # What to monitor
    patience=5,            # Stop after 5 epochs with no improvement
    restore_best_weights=True,  # Restore the best weights after stopping
    verbose=1              # Print a message when stopping
)

# Unfreeze the last 30 layers of the base model
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=25, 
                    batch_size=64,
                    callbacks=[early_stop])


model.evaluate(X_test, y_test, verbose=1)

model.save(reference_model_path.joinpath("resnet50_pneumonia.keras"))