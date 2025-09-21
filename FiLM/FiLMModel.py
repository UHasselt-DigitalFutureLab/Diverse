import os
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
from tensorflow import keras
from tensorflow.keras import layers
from FiLM.FiLMLayer import FiLMLayer

class FiLMModel:
    def __init__(self, referene_model_file, input_shape, name, random_seed, z_dim):
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.random_seed = random_seed
        self.reference_model_path = os.path.join("reference_models", referene_model_file)
        self.reference_model = self._load_trained_reference_model()
        if name == "mnist":
            self.model = self._rebuild_mnist_model()
        elif name == "resnet50_pneumonia":
            self.model = self._rebuild_resnet_model()
        elif name == "vgg16_cifar10":
           self.model = self.rebuild_vgg16_model()
        else:
           raise ValueError(f"Unknown model name: {name}. Supported names are 'mnist', 'resnet50_pneumonia', and 'vgg16_cifar10'.")
        self.sanity_check_rebuild_model()


    def sanity_check_rebuild_model(self):
        """
            Sanity check to ensure that the rebuilt model has the same weights as the reference model.
            This is necessary to ensure that the FiLMLayer is applied correctly.
        """
        for layer in self.model.layers:
            try:
                old_weights = self.reference_model.get_layer(name=layer.name).get_weights()
                new_weights = layer.get_weights()
                if not all([np.array_equal(old, new) for old, new in zip(old_weights, new_weights)]):
                    print("Weights mismatch for layer:", layer.name)
                    raise ValueError(f"Layer {layer.name} weights do not match!")
            except:
                pass

    
    def predict(self, x, z):
        """
            Predict using the latent vector z and the batch input x.
            x should be a batch of images with shape (batch_size, W, H).
            z should be a latent vector with shape (z_dim).
        """
        # Replicate the z value so it can be paired with every input in the batch
        z_batch = np.tile(z, (x.shape[0], 1))
        preds = self.model.predict([x, z_batch], verbose=0)
        return preds


    def _load_trained_reference_model(self):
        return keras.models.load_model(self.reference_model_path)
    

    def _transfer_weights_from_reference(self, new_model_blueprint: keras.Model) -> keras.Model:
        """
            Transfer weights from the reference model to the new model blueprint.
            This is necessary because the FiLMLayer is applied after the dense layers.
        """
        for layer in new_model_blueprint.layers:
            try:
                old_weights = self.reference_model.get_layer(name=layer.name).get_weights()
                layer.set_weights(old_weights)
            except:
                pass # Skip layers that are not in the reference model or FiLMLayers
        return new_model_blueprint
    

    def _freeze_base_layers(self, new_model_blueprint: keras.Model):
        """
            Freeze the base layers of the new model blueprint.
            This is necessary to ensure that the weights of the base layers are never updated.
        """
        for layer in new_model_blueprint.layers:
            if not isinstance(layer, FiLMLayer):
                layer.trainable = False
        return new_model_blueprint


    def _rebuild_mnist_model(self) -> keras.Model:
        """
            We need to rebuild the model architecture manually because the FiLMLayer has to be applied before the activation functions.
            This is a workaround to ensure the FiLMLayer is applied correctly before the activation functions.
            We assume that the FiLMLayer has to be applied after the dense layers.
        """
        x_in = keras.Input(shape=self.input_shape)
        z_in = keras.Input(shape=(self.z_dim,))

        x = layers.Flatten()(x_in)

        # Dense 1
        x = layers.Dense(128, activation=None, name="dense")(x)
        x = FiLMLayer(128, projection_dim=self.z_dim, random_seed=self.random_seed, name="film_1")(x, z_in)
        x = layers.Activation("elu")(x)

        # Dense 2
        x = layers.Dense(128, activation=None, name="dense_1")(x)
        x = FiLMLayer(128, projection_dim=self.z_dim, random_seed=self.random_seed, name="film_2")(x, z_in)
        x = layers.Activation("elu")(x)

        # Dense 3
        x = layers.Dense(128, activation=None, name="dense_2")(x)
        x = FiLMLayer(128, projection_dim=self.z_dim, random_seed=self.random_seed, name="film_3")(x, z_in)
        x = layers.Activation("elu")(x)

        # Output
        x = layers.Dense(10, activation="softmax", name="dense_3")(x)

        new_model_blueprint = keras.Model(inputs=[x_in, z_in], outputs=x)
        new_model_blueprint = self._transfer_weights_from_reference(new_model_blueprint)
        new_model_blueprint = self._freeze_base_layers(new_model_blueprint)
        return new_model_blueprint


    def _rebuild_resnet_model(self) -> keras.Model:

        ref: keras.Model = self.reference_model

        x_in = keras.Input(shape=ref.input_shape[1:], name="image_input")
        z_in = keras.Input(shape=(self.z_dim,), name="z_input")

        tensor_map = {ref.inputs[0]: x_in}

        def prev_layer_of_tensor(t):
            kh = getattr(t, "_keras_history", None)
            return getattr(kh, "layer", kh[0]) if kh is not None else None

        def is_relu(l: keras.layers.Layer) -> bool:
            if isinstance(l, layers.ReLU):
                return True
            if isinstance(l, layers.Activation):
                a = l.activation
                return a in (tf.nn.relu, keras.activations.relu) or a == "relu"
            return False

        def make_layer_like(orig: keras.layers.Layer) -> keras.layers.Layer:
            cfg = orig.get_config()
            cfg["name"] = orig.name
            return orig.__class__.from_config(cfg)

        film_idx = 0

        for orig in ref.layers:
            if isinstance(orig, keras.layers.InputLayer):
                continue

            # map inputs
            o_in = orig.input
            n_in = [tensor_map[t] for t in o_in] if isinstance(o_in, (list, tuple)) else tensor_map[o_in]

            # ReLU fed by BN -> FiLM between BN and ReLU
            if is_relu(orig):
                prev = prev_layer_of_tensor(orig.input)
                if isinstance(prev, layers.BatchNormalization):
                    film_idx += 1
                    y = FiLMLayer(
                        units=int(n_in.shape[-1]),
                        projection_dim=self.z_dim,
                        random_seed=self.random_seed,
                        name=f"film_{film_idx}",
                    )(n_in, z_in)
                    y = layers.Activation("relu", name=orig.name)(y)
                else:
                    new_layer = make_layer_like(orig)
                    y = new_layer(n_in)  # build
                    # (ReLU has no weights)
            elif isinstance(orig, layers.Add):
                add_in_new = n_in if isinstance(n_in, list) else [n_in]
                add_in_orig = orig.input if isinstance(orig.input, list) else [orig.input]
                new_add_inputs = []
                for nin, oin in zip(add_in_new, add_in_orig):
                    producer = prev_layer_of_tensor(oin)
                    if isinstance(producer, layers.BatchNormalization):
                        film_idx += 1
                        nin = FiLMLayer(
                            units=int(nin.shape[-1]),
                            projection_dim=self.z_dim,
                            random_seed=self.random_seed,
                            name=f"film_{film_idx}",
                        )(nin, z_in)
                    new_add_inputs.append(nin)
                y = layers.Add(name=orig.name)(new_add_inputs)
            else:
                new_layer = make_layer_like(orig)
                y = new_layer(n_in)  # build first
                # now it has weights -> safe to copy if any
                try:
                    ow = orig.get_weights()
                    if ow:
                        new_layer.set_weights(ow)
                except ValueError:
                    # shape/arity mismatch: nothing to copy
                    pass

            # record mapping
            if isinstance(orig.output, (list, tuple)):
                for to, tn in zip(orig.output, y):
                    tensor_map[to] = tn
            else:
                tensor_map[orig.output] = y

        out = tensor_map[ref.outputs[0]]
        model = keras.Model(inputs=[x_in, z_in], outputs=out, name="resnet50_with_film_per_conv")

        # freeze all non-FiLM layers
        model = self._freeze_base_layers(model)
        return model

    
    def rebuild_vgg16_model(self) -> keras.Model:
        ref: keras.Model = self.reference_model

        x_in = keras.Input(shape=ref.input_shape[1:], name="image_input")
        z_in = keras.Input(shape=(self.z_dim,), name="z_input")

        tensor_map = {ref.inputs[0]: x_in}

        def clone_like(orig):
            cfg = orig.get_config()
            cfg["name"] = orig.name
            return orig.__class__.from_config(cfg)

        film_idx = 0
        stop_after = "block5_pool"

        for orig in ref.layers:
            if isinstance(orig, keras.layers.InputLayer):
                continue

            o_in = orig.input
            n_in = [tensor_map[t] for t in o_in] if isinstance(o_in, (list, tuple)) else tensor_map[o_in]

            if isinstance(orig, layers.Conv2D):
                # Conv with linear activation
                cfg = orig.get_config()
                cfg["name"] = orig.name
                cfg["activation"] = None
                new_conv = layers.Conv2D.from_config(cfg)
                y = new_conv(n_in)
                new_conv.set_weights(orig.get_weights())

                # FiLM BEFORE activation
                film_idx += 1
                y = FiLMLayer(
                    units=cfg["filters"],
                    projection_dim=self.z_dim,
                    random_seed=self.random_seed,
                    name=f"film_{film_idx}",
                )(y, z_in)

                # Explicit ReLU (separate layer)
                y = layers.Activation("relu", name=f"{orig.name}_relu")(y)

            elif isinstance(orig, layers.MaxPooling2D):
                new_pool = clone_like(orig)
                y = new_pool(n_in)

            else:
                new_layer = clone_like(orig)
                y = new_layer(n_in)
                ow = getattr(orig, "get_weights", lambda: [])()
                if ow:
                    new_layer.set_weights(ow)

            # record mapping
            if isinstance(orig.output, (list, tuple)):
                for to, tn in zip(orig.output, y):
                    tensor_map[to] = tn
            else:
                tensor_map[orig.output] = y

            if orig.name == stop_after:
                break

        # classifier head (match ref dense sizes if present)
        x = tensor_map[ref.get_layer(stop_after).output]
        x = layers.Flatten(name="flatten")(x)

        d1_units = ref.get_layer("dense").units if "dense" in [l.name for l in ref.layers] else 256
        x = layers.Dense(d1_units, activation=None, name="dense")(x)

        film_idx += 1
        x = FiLMLayer(
            units=d1_units,
            projection_dim=self.z_dim,
            random_seed=self.random_seed,
            name=f"film_{film_idx}"
        )(x, z_in)
        x = layers.Activation("relu", name="dense_relu")(x)

        if any(l.name == "dropout" for l in ref.layers):
            x = layers.Dropout(ref.get_layer("dropout").rate, name="dropout")(x)
        else:
            x = layers.Dropout(0.5, name="dropout")(x)

        num_classes = ref.get_layer("dense_1").units if "dense_1" in [l.name for l in ref.layers] else 10
        out = layers.Dense(num_classes, activation="softmax", name="dense_1")(x)

        model = keras.Model(inputs=[x_in, z_in], outputs=out, name="vgg16_with_film")

        # copy dense weights if exist in reference
        for lname in ["dense", "dense_1"]:
            if lname in [l.name for l in ref.layers]:
                model.get_layer(lname).set_weights(ref.get_layer(lname).get_weights())

        model = self._freeze_base_layers(model)

        return model
