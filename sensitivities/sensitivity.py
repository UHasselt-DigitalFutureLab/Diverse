import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path

from utils.data_loader import Dataloader
from utils.experiment_parameters import get_experiment_parameters
from utils.reference_model import ReferenceModel
from utils.evaluation_metrics import total_variation_distance
from FiLM.FiLMModel import FiLMModel
from FiLM.FiLMLayer import FiLMLayer


def get_film_layers(film_model : FiLMModel) -> list:
    film_layers = []
    for layer in film_model.model.layers:
        if layer.__class__.__name__ == "FiLMLayer":
            film_layers.append(layer)

    if not film_layers:
        raise ValueError("No FiLM layers found!")

    return film_layers


def gamma_beta_from_layer(layer : FiLMLayer, z_row : np.ndarray):
    # Convert z to tensor, with the correct shaping to match a batch of one vector (1, d)
    z = tf.convert_to_tensor(z_row[np.newaxis, :], dtype=tf.float32)
    g, b = layer.get_gamma_beta(z)   # expected shapes: (1, d), (1, d)
    return g.numpy().squeeze(), b.numpy().squeeze()


def set_all_gates(model, value: float = 1.0):
    for layer in get_film_layers(model):
        layer.set_gate(value)


def layer_sensitivity_delta(film_model: FiLMModel, reference_model: ReferenceModel, x_val, z):
    """
    Calculate for one Rashomon member
        1) compute baseline disagreement vs base model (all gates on)
        2) turn off each FiLM layer separately
        3) record ΔD = D_all_on - D_off(layer)
    """
    film_layers = get_film_layers(film_model)

    # reference model predictions
    ref_probs, _ = reference_model.get_probabilties_and_accuracy("val")

    # baseline: all FiLMs on
    set_all_gates(film_model, 1.0)
    probs_all_on = film_model.predict(x_val, z)
    cand_labels = np.argmax(probs_all_on, axis=1)
    tvd_all_on = total_variation_distance(probs_all_on, ref_probs).mean()

    disagreement = cand_labels != np.argmax(ref_probs, axis=1)
    disagreement_mean_all_on = np.mean(disagreement)

    results = []
    for layer in film_layers:
        layer.set_gate(0.0)
        probs_off = film_model.predict(x_val, z)
        cand_labels = np.argmax(probs_off, axis=1)
        disagreement = cand_labels != np.argmax(ref_probs, axis=1)
        disagreement_mean_off = np.mean(disagreement)
        delta = disagreement_mean_all_on - disagreement_mean_off
        results.append((layer.name, float(delta)))
        layer.set_gate(1.0)
    set_all_gates(film_model, 1.0)
    results.sort(key=lambda t: t[1], reverse=True)
    return results, tvd_all_on


def aggregate_sensitivity(film_model: FiLMModel, reference_model: ReferenceModel, x_val, Z):
    rows = []
    members = Z
    for j, z in enumerate(members):
        print(f"Starting model {j} out of {len(members)}", flush=True)
        sens, tvd_base = layer_sensitivity_delta(film_model, reference_model, x_val, z)
        for lname, delta in sens:
            rows.append({"member_id": j, "layer": lname,
                         "deltaD": delta, "D_all_on": tvd_base})
    df = pd.DataFrame(rows)
    summary = (df.groupby("layer", as_index=False)
                 .agg(mean_deltaD=("deltaD", "mean"),
                      iqr_deltaD=("deltaD", lambda s: np.subtract(*np.percentile(s, [75,25]))))
                 .sort_values("mean_deltaD", ascending=False))
    return df, summary


def file_name_to_parameters(file_name : str) -> tuple[str, float]:
    # the file structure can have the following:
    ## model_evaluations_z0_seed_seed_#_sigma_0.#.csv
    ## model_evaluations_z0_zeros_sigma_0.#.csv
    ## model_evaluations_z0_ones_sigma_0.#.csv
    # Split on _
    parts = file_name.split("_")
    if len(parts) == 6:
        # We have zeros or ones
        _, _, _, zeros_ones, _, sigma_csv = parts
        sigma_value = float(sigma_csv.removesuffix(".csv"))
        return zeros_ones, sigma_value
    # We have a z_seed
    _, _, _, _, z_seed, _, sigma_csv = parts
    sigma_value = float(sigma_csv.removesuffix(".csv"))
    return z_seed, sigma_value
    

def is_correct_file(file: str, a_z_seed: str, a_sigma: float) -> bool:
    z_seed, sigma_value = file_name_to_parameters(file)
    return z_seed == a_z_seed and sigma_value == a_sigma


def search_folder_for_file(search_folder: str, a_z_seed: str, a_sigma: float) -> str:
    # Get the matching file, if there isn't one fill it with None
    file = next((f for f in os.listdir(search_folder) if is_correct_file(f, a_z_seed, a_sigma)), None)
    if not file:
        raise FileNotFoundError(f"File with the given parameters does not exist: sigma {a_sigma} z_seed {a_z_seed}")
    return file


def handle_cli_args():
    """ Handle command line arguments for the CMA-ES script. """
    parser = argparse.ArgumentParser(description="Run CMA-ES for different z_0 vectors and sigma values and different experiments.")
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for the Rashomon requirement", choices=[0.01, 0.02, 0.03, 0.04, 0.05])
    parser.add_argument("--z_dim", type=int, required=True, choices=[2, 4, 8, 16, 32, 64], help="Dimensionality of the z vector.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to use", choices=["mnist", "resnet50_pneumonia", "vgg16_cifar10"])
    parser.add_argument("--sigma", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5], help="Initial standard deviation for CMA-ES", required=True)
    parser.add_argument("--lambda_val", type=float, required=False, default=0.5, help="Lambda value used as a mixing weight for hard and soft disagreement", choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--z_seed", type=str, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "zeros", "ones"], required=True, help="Z_vector initialization seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = handle_cli_args()
    epsilon = args.epsilon
    z_dim = args.z_dim
    model_type = args.model_type
    lambda_val = args.lambda_val
    sigma = args.sigma
    z_seed = args.z_seed

    params = get_experiment_parameters(model_type)
    dataloader = Dataloader(test_x_path=params["test_x_path"], test_y_path=params["test_y_path"], val_x_path=params["val_x_path"], val_y_path=params["val_y_path"])
    output_path = Path(f"sensitivity_results/{model_type}/epsilon_{epsilon}_lambda_{lambda_val}/sigma_{sigma}/z_seed_{z_seed}")
    output_path.mkdir(parents=True, exist_ok=True)
    input_folder = f"cma_evaluations/{model_type}/epsilon_{epsilon}_lambda_{lambda_val}/z_{z_dim}/model_evaluations/"

    try:
        input_csv = search_folder_for_file(input_folder, z_seed, sigma)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    df = pd.read_csv(os.path.join(input_folder, input_csv))

    # Get z_column out of .csv, we dont want the z_folder (z_seed)
    z_cols = [col for col in df.columns if col != "z_folder" and col.startswith("z_")]
    # Filter out found z_values so only the Rashomon set is included (based on validation performance)
    df = df.loc[df["is_in_rashomon_set"]]
    # Get the z_values of that Rashomon set
    z_values = df[z_cols].values


    film_model = FiLMModel(referene_model_file=params["reference_model"], input_shape=params["input_shape"], name=model_type, random_seed=42, z_dim=z_dim)
    ref_model = ReferenceModel(film_model, dataloader=dataloader, z_dim=z_dim)
    ref_probs, ref_acc = ref_model.get_probabilties_and_accuracy("val")

    temp_df, temp_summary = aggregate_sensitivity(film_model, ref_model, dataloader.x_val, z_values)
    temp_df.to_csv(output_path / "layer_sensitivity.csv")
    temp_summary.to_csv(output_path / "sensitivity_summary.csv")