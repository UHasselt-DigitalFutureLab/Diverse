import os
import argparse
import numpy as np
import pandas as pd
from utils.evaluation_metrics import rashomon_check, true_class_probablities, ambiguity, discrepancy, viable_prediction_range, rashomon_capacity
from baselines.retraining_helper import get_resnet_data, train_and_evaluate_one_resnet_model, get_vgg_data, train_and_evaluate_one_vgg_model, get_mnist_data, train_and_evaluate_one_mnist_model

def to_int_labels(y):
    """ Convert one-hot or int labels to int labels."""
    y = np.asarray(y)
    if y.ndim == 2:            # one-hot
        return y.argmax(axis=1)
    elif y.ndim == 1:          # already int labels
        return y
    else:
        raise ValueError(f"Unexpected y shape: {y.shape}")

def get_rashomon_metrics(rashomon_probs, rashomon_labels, y_true, ref_preds):
    """
    Compute Rashomon set metrics.
    """
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


def get_resnet_ref():
    """Get reference results for ResNet model.
     Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_resnet_data(42)
    (val_loss,
    test_loss,
    y_probs,
    y_preds,
    y_true) = train_and_evaluate_one_resnet_model(42, x_train, y_train, x_val, y_val, x_test, y_test)
    return val_loss, test_loss, y_probs, y_preds, y_true


def get_vgg_ref():
    """Get reference results for VGG model.
     Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_vgg_data(45)
    (val_loss,
    test_loss,
    y_probs,
    y_preds,
    y_true) = train_and_evaluate_one_vgg_model(45, x_train, y_train, x_val, y_val, x_test, y_test)
    return val_loss, test_loss, y_probs, y_preds, y_true

def get_mnist_ref():
    """Get reference results for MNIST model.
     Returns:
        Tuple of (val_loss, test_loss, y_probs, y_preds, y_true).
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data(42)
    (val_loss,
    test_loss,
    y_probs,
    y_preds,
    y_true) = train_and_evaluate_one_mnist_model(42, x_train, y_train, x_val, y_val, x_test, y_test)
    return val_loss, test_loss, y_probs, y_preds, y_true

def get_data_from_chunk(npz_folder, epsilon:float, models: int, ref_val_loss: float, ref_test_loss: float, y_true, ref_preds):
    """
    Process a chunk of .npz files to compute Rashomon set metrics and write to CSV.
    Args:
        npz_folder (str): Folder containing .npz files.
        epsilon (float): Tolerance for Rashomon set inclusion.
        models (int): Number of models to evaluate.
        ref_val_loss (float): Reference validation loss.
        ref_test_loss (float): Reference test loss.
        y_true (np.ndarray): True labels.
        ref_preds (np.ndarray): Reference model predictions.
    """
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith(".npz")]
    rashomon_probs = []
    rashomon_preds = []
    train_times = []
    # Read all files and check if loss is in rashomon_check
    for i, npz_file in enumerate(npz_files):
        if i == models:
            break
        data = np.load(os.path.join(npz_folder, npz_file))
        seeds      = data["seeds"]
        val_loss   = data["val_loss"]
        test_loss  = data["test_loss"]
        train_time = data["train_time"]
        test_preds = data["test_preds"]
        test_probs = data["test_probs"]
        train_times.append(train_time)
        # Check rashomon set
        val_in_rashomon = rashomon_check(val_loss, ref_val_loss, epsilon=epsilon)
        if val_in_rashomon:
            test_in_rashomon = rashomon_check(test_loss, ref_test_loss, epsilon=epsilon)
            if test_in_rashomon:
                rashomon_probs.append(test_probs)
                rashomon_preds.append(test_preds)
    rashomon_probs = np.array(rashomon_probs)
    rashomon_preds = np.array(rashomon_preds)
    rashomon_probs = rashomon_probs.squeeze(1)
    rashomon_preds = rashomon_preds.squeeze(1)
    amb, disc, vpr_width_mean, rc_mean = get_rashomon_metrics(rashomon_probs, rashomon_preds, y_true, ref_probs)
    results = {
        "total_models_evaluated": models,
        "epsilon": epsilon,
        "training_time": float(np.array(train_times).sum()),
        "amb": amb,
        "disc": disc,
        "vpr_width_mean": vpr_width_mean,
        "rc_mean": rc_mean,
        "num_models_in_rashomon": rashomon_probs.shape[0]
    }
    df = pd.DataFrame([results])
    os.makedirs(os.path.join(npz_folder, f"epsilon_{epsilon}"), exist_ok=True)
    out_csv = os.path.join(npz_folder, f"epsilon_{epsilon}", f"{models}_rashomon_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved rashomon metrics to {out_csv}")


def handle_cli_args():
    """Handle command line arguments."""
    parser = argparse.ArgumentParser(description="Train models with different seeds.")
    parser.add_argument("--model", type=str, choices=["resnet", "vgg", "mnist"], required=True,
                        help="Type of model to train: 'resnet', 'vgg', or 'mnist'.")
    parser.add_argument("--search_budget", type=int, choices=[162, 320, 640, 1284, 2562, 5120], required=True,
                        help="Maximum number of models to evaluate.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_cli_args()
    model_type = args.model
    if model_type == "mnist":
        ref_val_loss, ref_test_loss, ref_probs, ref_preds, y_true = get_mnist_ref()
    elif model_type == "resnet":
        ref_val_loss, ref_test_loss, ref_probs, ref_preds, y_true = get_resnet_ref()
    elif model_type == "vgg":
        ref_val_loss, ref_test_loss, ref_probs, ref_preds, y_true = get_vgg_ref()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
        print(f"Processing epsilon: {epsilon}")
        for models in [args.search_budget]:
            npz_folder = f"baseline_evaluations/retraining/retraining_{model_type}"
            get_data_from_chunk(npz_folder, epsilon, models, ref_val_loss, ref_test_loss, y_true, ref_preds)
