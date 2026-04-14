import os
import random
import keras
import tensorflow as tf
from keras import layers
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from init.custom_layers import Patches, PatchEncoder
seed = 45

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

num_classes = 10
input_shape = (32, 32, 3)
reference_model_path = Path('reference_models')
dataset_model_path = Path('datasets/cifar10_vit')

reference_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
dataset_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

reference_model_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 75  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
] # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.Normalization(name="input_normalization"),
        layers.Resizing(image_size, image_size, name="input_resizing"),
    ],
    name="data_augmentation",
)





(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
# save dataset
np.save(dataset_model_path.joinpath('x_train.npy'), x_train)
np.save(dataset_model_path.joinpath('y_train.npy'), y_train)
np.save(dataset_model_path.joinpath('x_val.npy'), x_val)
np.save(dataset_model_path.joinpath('y_val.npy'), y_val)
np.save(dataset_model_path.joinpath('x_test.npy'), x_test)
np.save(dataset_model_path.joinpath('y_test.npy'), y_test)


print("Training data shape:", x_train.shape)
print("Validation data shape:", x_val.shape)
print("Test data shape:", x_test.shape)

def mlp(x, hidden_units, dropout_rate, block_name):
    """
    block_name: string prefix, e.g. "encoder_block_0_mlp" or "mlp_head"
    """
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=None,
            name=f"{block_name}_dense_{i}",
        )(x)

        x = layers.Activation(keras.activations.gelu, name=f"{block_name}_gelu_{i}")(x)
        x = layers.Dropout(
            dropout_rate,
            name=f"{block_name}_dropout_{i}",
        )(x)
    return x

    
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape, name="image")
    # Augment data.
    print("Input shape:", inputs.shape)
    augmented = data_augmentation(inputs)
    print("Augmented shape:", augmented.shape)
    # Create patches.
    patches = Patches(patch_size=patch_size, name="patches")(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, name="patch_encoder")(patches)

    # Create multiple layers of the Transformer block.
    for block_idx in range(transformer_layers):
        block_name = f"encoder_block_{block_idx}"
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{block_name}_layer_norm_1")(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name=f"{block_name}_multi_head_attention"
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add(name=f"{block_name}_skip_connection_1")([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"{block_name}_layer_norm_2")(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, block_name=f"{block_name}_mlp")
        # Skip connection 2.
        encoded_patches = layers.Add(name=f"{block_name}_skip_connection_2")([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name="representation_layer_norm")(encoded_patches)
    representation = layers.Flatten(name="flatten")(representation)
    representation = layers.Dropout(0.5, name="dropout")(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5, block_name="mlp_head")
    # Classify outputs.
    logits = layers.Dense(num_classes, name="dense_1", activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    # Export model
    model.save(reference_model_path.joinpath('vision_transformer.keras'))
    return history


if __name__ == "__main__":
    vit_classifier = create_vit_classifier()
    history = run_experiment(vit_classifier)