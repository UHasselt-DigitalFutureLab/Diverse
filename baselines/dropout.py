import random
import time
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from utils.evaluation_metrics import rashomon_check, true_class_probablities, ambiguity, discrepancy, viable_prediction_range, rashomon_capacity

SEED = 1234
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class ControlledDropoutWrapper(layers.Wrapper):
    """
    Wraps a layer and applies controlled Bernoulli or Gaussian dropout
    to the *outputs* of the wrapped layer on every call (train & infer),
    matching the PyTorch-hook behavior (training=True).
    """
    def __init__(self, layer, method="bernoulli", rate=0.0, name=None, **kwargs):
        if not isinstance(layer, layers.Layer):
            raise ValueError("ControlledDropoutWrapper expects a Keras Layer.")
        super().__init__(layer, name=name, **kwargs)
        if method not in ("bernoulli", "gaussian"):
            raise ValueError(f"Unknown method '{method}'")
        self.method = method
        self.rate = tf.Variable(rate, trainable=False, dtype=tf.float32, name="drop_rate")

    def build(self, input_shape):
        # Make sure the inner layer is built; preserves its weights
        if not self.layer.built:
            self.layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        y = self.layer(inputs, **kwargs)
        p = tf.clip_by_value(self.rate, 0.0, 1.0)

        # Always apply, regardless of Keras training phase, to match hooks.
        if self.method == "bernoulli":
            # tf.nn.dropout scales by (1-p); identical to torch.nn.functional.dropout(training=True)
            return tf.nn.dropout(y, rate=p)
        else:  # gaussian: y * (N(0,1) * p + 1)
            noise = tf.random.normal(tf.shape(y), dtype=y.dtype)
            return y * (noise * p + 1.0)

    def get_config(self):
        base = super().get_config()
        base.update({
            "method": self.method,
            "rate": float(self.rate.numpy()) if hasattr(self.rate, "numpy") else 0.0,
        })
        return base

    @property
    def p(self):
        return float(self.rate.numpy())

    @p.setter
    def p(self, value: float):
        self.rate.assign(float(value))


def _should_wrap(layer):
    if isinstance(layer, (layers.Dropout, layers.GaussianDropout)):
        return False
    return isinstance(layer, (layers.Conv2D, layers.Dense))


def add_dropout(model: tf.keras.Model, method: str = "bernoulli", initial_p: float = 0.0) -> tf.keras.Model:
    """
    Clone `model`, wrapping each Conv2D/Dense with ControlledDropoutWrapper(method, initial_p).
    All original weights are preserved.
    """
    def clone_fn(layer):
        if _should_wrap(layer):
            wrapped = ControlledDropoutWrapper(
                layer.__class__.from_config(layer.get_config()),
                method=method,
                rate=initial_p,
                name=(layer.name + f"_{method}_cdrop")
            )
            return wrapped
        # Return a fresh layer of the same config for others
        return layer.__class__.from_config(layer.get_config())

    # Clone architecture and load weights
    cloned = models.clone_model(model, clone_function=clone_fn)
    cloned.build(model.input_shape)
    cloned.set_weights(model.get_weights())
    return cloned


def change_dropout_rate(model: tf.keras.Model, p: float) -> tf.keras.Model:
    """
    Set dropout rate `p` on all ControlledDropoutWrapper instances.
    """
    for layer in model.layers:
        # Dive into nested models/Functional blocks too
        for sub in layer._flatten_layers(include_self=True, recursive=True):
            if isinstance(sub, ControlledDropoutWrapper):
                sub.p = p
    return model

def get_dropout_sweep(method: str, ndrp: int):
    """
    Returns the list of dropout rates used in the baseline for a given method.
    method: "bernoulli" or "gaussian"
    ndrp: number of points in the sweep (same as args.ndrp)
    """
    if method == "bernoulli":
        drp_max_ratio = 0.008
    elif method == "gaussian":
        drp_max_ratio = 0.1
    else:
        raise ValueError("method must be 'bernoulli' or 'gaussian'")
    return np.linspace(0.0, drp_max_ratio, ndrp)

def get_logits(model, x):
    y = model(x, training=False)  # wrapper keeps dropout on; BN stays in inference
    return tf.convert_to_tensor(y).numpy()

def evaluate_dropout_models(model, x_test, y_test, method: str, ndrp: int, drp_nmodel: int, epsilon, test_reference_loss : float):
    drp_list = get_dropout_sweep(method, ndrp)
    change_dropout_rate(model, 0.0)
    test_probs = []
    test_preds = []
    timings = []
    for _, p in enumerate(drp_list):
        change_dropout_rate(model, float(p))
        for _ in range(drp_nmodel):
            # Use Keras evaluate (BN in inference; dropout still active due to wrapper)
            start_time = time.time()
            loss, _ = model.evaluate(x_test, y_test, verbose=0)
            end_time = time.time()
            timings.append(end_time - start_time)
            test_in_rashomon_set = rashomon_check(loss, test_reference_loss, epsilon=epsilon)
            if test_in_rashomon_set:
                probs = model.predict(x_test, verbose=0)
                test_probs.append(probs)
                test_preds.append(np.argmax(probs, axis=1))

    return ndrp * drp_nmodel, test_probs, test_preds, sum(timings)

def to_int_labels(y):
    y = np.asarray(y)
    if y.ndim == 2:            # one-hot
        return y.argmax(axis=1)
    elif y.ndim == 1:          # already int labels
        return y
    else:
        raise ValueError(f"Unexpected y shape: {y.shape}")

def get_rashomon_metrics(rashomon_probs, rashomon_labels, y_true, ref_preds):
    rashomon_scores = rashomon_probs
    rashomon_decisions = rashomon_labels
    y_true = to_int_labels(y_true)
    # Probability-based
    score_y = true_class_probablities(rashomon_scores, y_true)
    vpr_min, vpr_max, vpr_width = viable_prediction_range(score_y) # 0 is no viable prediction range, 1 is all predictions are viable
    #rc = rashomon_capacity(rashomon_scores, dataloader.y_test)  # scalar, the number of models in the Rashomon set that are viable
    # Decision-based 
    amb = ambiguity(rashomon_decisions, np.argmax(ref_preds, axis=1)) # scalar, that represents the % of test samples, where at least one model disagrees with the reference model
    disc = discrepancy(rashomon_decisions, np.argmax(ref_preds, axis=1)) # scalar, there is at least one model in the Rashomon set that disagrees from the base model on scalar% of the test set
    # score-based: Rashomon capacity (exact match)
    rc_per_sample = rashomon_capacity(rashomon_scores)  # (N,) bits
    rc_effective = 2 ** rc_per_sample  # (N,) effective number of models
    rc_mean = float(rc_effective.mean())
    return amb, disc, vpr_width.mean(), rc_mean

def handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mnist", "vgg16_cifar10", "resnet50_pneumonia"], required=True,
                        help="Which reference model to use.")
    parser.add_argument("--epsilon", type=float, choices=[0.01, 0.02, 0.03, 0.04, 0.05], required=True,
                        help="Rashomon set threshold (fraction of reference loss).")
    parser.add_argument("--search_budget", type=int, choices=[162, 320, 640, 1284, 2562, 5120], required=True,
                        help="Total number of models to evaluate (approx).")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = handle_cli_args()
    # 1. Load your pretrained model
    if args.model == "mnist":
        base = keras.models.load_model("reference_models/mnist_base.keras")
        x_test = np.load("datasets/mnist/x_test.npy")
        y_test = np.load("datasets/mnist/y_test.npy")
        wrapped = add_dropout(base, method="gaussian", initial_p=0.0)
        wrapped.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    elif args.model == "vgg16_cifar10":
        base = keras.models.load_model("reference_models/vgg16_cifar10.keras")
        x_test = np.load("datasets/cifar10_vgg16/x_test.npy")
        y_test = np.load("datasets/cifar10_vgg16/y_test.npy")
        wrapped = add_dropout(base, method="gaussian", initial_p=0.0)
        wrapped.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    elif args.model == "resnet50_pneumonia":
        base = keras.models.load_model("reference_models/resnet50_pneumonia.keras")
        x_test = np.load("datasets/pneumonia_mnist/x_test_normalized.npy")
        y_test = np.load("datasets/pneumonia_mnist/y_test_onehot.npy")
        wrapped = add_dropout(base, method="gaussian", initial_p=0.0)
        wrapped.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=['accuracy'])
    else:
        raise ValueError("Unknown model")

    # 2. Adjust dropout rate as needed
    change_dropout_rate(wrapped, 0)

    # 3. Evaluate with dropout active at 0 dropout rate to get reference loss
    results = wrapped.evaluate(x_test, y_test, return_dict=True)
    ref_loss = results['loss']
    ref_preds = wrapped.predict(x_test)

    for m in [args.search_budget]:
        for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
            change_dropout_rate(wrapped, 0)
            n_models_searched, rashomon_probs, rashomon_preds, sum_timings = evaluate_dropout_models(wrapped, x_test, y_test, method="gaussian", ndrp=5, drp_nmodel=math.ceil(m / 5), epsilon=epsilon, test_reference_loss=ref_loss)
            rashomon_probs = np.stack(rashomon_probs, axis=0)
            rashomon_preds = np.stack(rashomon_preds, axis=0)
            if len(rashomon_probs):
                amb, disc, vpr_mean, rc_mean = get_rashomon_metrics(rashomon_probs, rashomon_preds, y_test, ref_preds=ref_preds)
                data_row = {
                    "dataset": args.model,
                    "method": "gaussian_dropout",
                    "total_models": n_models_searched,
                    "time_to_search": sum_timings,
                    "epsilon": epsilon,
                    "rashomon_models": len(rashomon_probs),
                    "rashomon_ratio": len(rashomon_probs) / n_models_searched,
                    "ambiguity": amb,
                    "discrepancy": disc,
                    "vpr": vpr_mean,
                    "rashomon_capacity": rc_mean
                    }
                df = pd.DataFrame([data_row])
                out_path = Path(f"baseline_evaluations/dropout/{args.model}/epsilon{epsilon}")
                out_path.mkdir(parents=True, exist_ok=True) 
                df.to_csv(f"{out_path}/rashomon_metrics_gaussian_dropout_{m}_models.csv", index=False)
            else:
                print("No models in Rashomon set", flush=True)
