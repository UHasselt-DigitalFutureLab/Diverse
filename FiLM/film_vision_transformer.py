import random
import numpy as np
import tensorflow as tf
random.seed(45)
np.random.seed(45)
tf.random.set_seed(45)
from FiLM.FiLMLayer import FiLMLayer
import keras
from keras import layers, ops
from init.custom_layers import Patches, PatchEncoder



learning_rate = 0.001
weight_decay = 0.0001
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


def mlp_with_film(x, z_in, z_dim, hidden_units, dropout_rate, block_name, random_seed):
    """
    MLP block used in the FiLM-wrapped ViT.
    - Dense layers keep EXACTLY the same names as in the reference ViT:
        encoder_block_{k}_mlp_dense_{i}, mlp_head_dense_{i}
    - We insert FiLM right after each Dense.
    - GELU + Dropout follow as usual.
    """
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=None,
            name=f"{block_name}_dense_{i}",
        )(x)
        x = FiLMLayer(units=units, projection_dim=z_dim, random_seed=random_seed, name=f"{block_name}_film_{i}")(x, z_in)
        x = layers.Activation(keras.activations.gelu, name=f"{block_name}_gelu_{i}")(x)
        x = layers.Dropout(
            dropout_rate,
            name=f"{block_name}_dropout_{i}",
        )(x)
    return x


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


def create_vit_classifier_film(z_dim, random_seed, input_shape=(32,32,3), num_classes=10):
    """
    FiLM-enabled Vision Transformer.

    Inputs:
      - image: same input_shape as reference (e.g., (32, 32, 3))
      - z: FiLM latent vector of dimension z_dim

    Architecture:
      - identical to create_vit_classifier() for all base layers
      - FiLM only after:
          encoder_block_{0..7}_mlp_dense_{0,1}
          mlp_head_dense_{0,1}
    """
    # Inputs
    z_in = keras.Input(shape=(z_dim,), name="z_input")
    img_in = keras.Input(shape=input_shape, name="image")
    # Augment data.
    data_augmentation = keras.Sequential(
    [
        layers.Normalization(name="input_normalization"),
        layers.Resizing(image_size, image_size, name="input_resizing"),
        # layers.RandomFlip("horizontal"),
        # layers.RandomRotation(factor=0.02),
        # layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",)

    augmented = data_augmentation(img_in)
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
        if block_idx == 7:
            x3 = mlp_with_film(x3, z_in, z_dim, hidden_units=transformer_units, dropout_rate=0.1, block_name=f"{block_name}_mlp", random_seed=random_seed)
        else:
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, block_name=f"{block_name}_mlp")
        # Skip connection 2.
        encoded_patches = layers.Add(name=f"{block_name}_skip_connection_2")([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name="representation_layer_norm")(encoded_patches)
    representation = layers.Flatten(name="flatten")(representation)
    representation = layers.Dropout(0.5, name="dropout")(representation)
    # Add MLP.
    #features = mlp_with_film(representation, z_in, z_dim, hidden_units=mlp_head_units, dropout_rate=0.5, block_name="mlp_head", random_seed=random_seed)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5, block_name="mlp_head")
    # Classify outputs.
    logits = layers.Dense(num_classes, name="dense_1", activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=[img_in, z_in], outputs=logits)
    return model
