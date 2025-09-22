import random
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from medmnist import PneumoniaMNIST
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_mnist_data(seed: int):
    """
    Load and preprocess MNIST data.
    Args:
        seed (int): Random seed for reproducibility.
    Returns:
        Tuple of preprocessed (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Validation split from training data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)

    # Normalize input data
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_vgg_data(seed: int):
    """
    Load and preprocess CIFAR-10 data for VGG16.
    Args:
        seed (int): Random seed for reproducibility.
    Returns:
        Tuple of preprocessed (x_train, y_train, x_val, y_val, x_test, y_test).
    """
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
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_resnet_data(seed: int):
    """
    Load and preprocess PneumoniaMNIST data for ResNet50.
    Args:
        seed (int): Random seed for reproducibility.
    Returns:
        Tuple of preprocessed (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
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

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_mnist_blueprint_model():
    """
    Define and compile a simple feedforward neural network for MNIST.
    Returns:
        Compiled Keras model.
    """
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
    return model


def train_and_evaluate_one_mnist_model(seed: int, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Train and evaluate one MNIST model with the given seed.
    Args:
        seed (int): Random seed for reproducibility.
        x_train, y_train: Training data and labels.
        x_val, y_val: Validation data and labels.
        x_test, y_test: Test data and labels.
    Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    model = None
    history = None
    try:
        model = get_mnist_blueprint_model()
        history = model.fit(
            x_train, y_train,
            epochs=50, batch_size=64,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            validation_data=(x_val, y_val), verbose=0
        )
        val_loss = float(history.history["val_loss"][-1])
        test_loss, _ = model.evaluate(x_test, y_test, verbose=0)
        y_probs = model.predict(x_test, verbose=0)          # (N_test, 10) float32
        y_preds = np.argmax(y_probs, axis=1)                # (N_test,) int64
        y_true  = np.argmax(y_test, axis=1)                 # (N_test,) int64
        # Cast down BEFORE returning to reduce memory pressure in caller
        return val_loss, float(test_loss), y_probs.astype(np.float16, copy=False), y_preds.astype(np.int16, copy=False), y_true.astype(np.int16, copy=False)
    finally:
        # Make sure TF state is released each iteration (you do this already in VGG) :contentReference[oaicite:1]{index=1}
        try:
            del model, history
        except NameError:
            pass
        K.clear_session()
        gc.collect()


def train_and_evaluate_one_vgg_model(seed: int, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Train and evaluate one VGG16 model with the given seed.
    Args:
        seed (int): Random seed for reproducibility.
        x_train, y_train: Training data and labels.
        x_val, y_val: Validation data and labels.
        x_test, y_test: Test data and labels.
    Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    try:
        y_train = np.asarray(y_train).squeeze().astype('int32')
        y_val   = np.asarray(y_val).squeeze().astype('int32')
        y_test  = np.asarray(y_test).squeeze().astype('int32')
        # Build the base VGG16 model (no top) and freeze its weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        base_model.trainable = False

        x = layers.Flatten()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=outputs)

        # Compile the model for head training
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Early stopping callback for head training
        early_stop_head = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the head
        model.fit(
            x_train, y_train,
            epochs=30,
            batch_size=64,
            validation_data=(x_val, y_val),
            callbacks=[early_stop_head],
            verbose=0
        )

        # Unfreeze last convolutional block for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers:
            if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
                layer.trainable = True
            else:
                layer.trainable = False

        # Re-compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Early stopping callback for fine-tuning
        early_stop_finetune = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 9. Fine-tune the model
        history = model.fit(
            x_train, y_train,
            epochs=30,
            batch_size=64,
            validation_data=(x_val, y_val),
            callbacks=[early_stop_finetune],
            verbose=0
        )

        # 10. Evaluate on the test set
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
        print(f"Seed: {seed}, Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}, val_acc: {history.history['val_accuracy'][-1]:.4f}")
        val_loss = history.history["val_loss"][-1]
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        y_probs = model.predict(x_test, verbose=0)  # softmax probabilities
        y_preds = np.argmax(y_probs, axis=1)                   # predicted labels
        y_true = y_test         
        return val_loss, test_loss, y_probs, y_preds, y_true
    finally:
        try:
            del model, base_model, x, outputs
        except:
            pass
        K.clear_session()
        gc.collect()

def train_and_evaluate_one_resnet_model(seed: int, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Train and evaluate one ResNet50 model with the given seed.
    Args:
        seed (int): Random seed for reproducibility.
        x_train, y_train: Training data and labels.
        x_val, y_val: Validation data and labels.
        x_test, y_test: Test data and labels.
    Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
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

        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=25,
                            batch_size=64,
                            callbacks=[early_stop], verbose=0)


        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        val_loss = history.history["val_loss"][-1]
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        y_probs = model.predict(x_test, verbose=0)  # softmax probabilities
        y_preds = np.argmax(y_probs, axis=1)                   # predicted labels
        y_true = y_test         
        return val_loss, test_loss, y_probs, y_preds, y_true

    finally:
        try:
            del model, x, history
        except:
            pass
        K.clear_session()
        gc.collect()