import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import random

seed = 45

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
reference_model_path = Path('reference_models')
dataset_model_path = Path('datasets/cifar10_vgg16')

reference_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
dataset_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Validation split from training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
# 2. Preprocess inputs and labels
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)


# 3. Build the base VGG16 model (no top) and freeze its weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)




# 4. Compile the model for head training
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback for head training
early_stop_head = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. Train the head
model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stop_head]
)

# 6. Unfreeze last convolutional block for fine-tuning
base_model.trainable = True
for layer in base_model.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True
    else:
        layer.trainable = False

# 7. Re-compile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback for fine-tuning
early_stop_finetune = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 8. Fine-tune the model
model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stop_finetune]
)

# 9. Evaluate on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")


# 10. Save the model
model.save(reference_model_path / 'vgg16_cifar10.keras')

# 11. Save the preprocessed data
np.save(dataset_model_path.joinpath("x_train.npy"), x_train)
np.save(dataset_model_path.joinpath("x_test.npy"), x_test)
np.save(dataset_model_path.joinpath("x_val.npy"), x_val)    
np.save(dataset_model_path.joinpath("y_train.npy"), y_train)
np.save(dataset_model_path.joinpath("y_test.npy"), y_test)
np.save(dataset_model_path.joinpath("y_val.npy"), y_val)

