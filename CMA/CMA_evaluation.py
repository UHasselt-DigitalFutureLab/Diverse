import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from utils.experiment_parameters import get_experiment_parameters
from utils.data_loader import Dataloader
from utils.reference_model import ReferenceModel
from utils.evaluation_metrics import rashomon_check, true_class_probablities, ambiguity, discrepancy, viable_prediction_range, rashomon_capacity
from FiLM.FiLMModel import FiLMModel


class CMAEvaluator:
    def __init__(self, seed, epsilon, model_type, z_dim, experiment_folder):
        np.random.seed(seed)
        self.seed = seed
        self.epsilon = epsilon
        self.model_type = model_type
        self.z_dim = z_dim
        self.experiment_folder = experiment_folder
        self.experiment_parameters = get_experiment_parameters(self.model_type)
        self.FiLM_model = FiLMModel(self.experiment_parameters["reference_model"], self.experiment_parameters["input_shape"], self.model_type, self.seed, self.z_dim)
        self.dataloader = Dataloader(test_x_path=self.experiment_parameters["test_x_path"], test_y_path=self.experiment_parameters["test_y_path"])
        self.sigmas = [s for s in os.listdir(self.experiment_folder)]
        self.sigmas = sorted(self.sigmas, key=lambda x: float(x.split("_")[1]))
        self.experiment_dfs = self.load_experiment_data()
        self.ref_preds, self.ref_acc, self.ref_loss = self._get_reference_results(subset="test")
        self.output_folder = Path("cma_evaluations") / f"{self.model_type}" / f"epsilon_{self.epsilon}" / f"z_{self.z_dim}"
        (self.output_folder / "model_evaluations").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "rashomon_metrics").mkdir(parents=True, exist_ok=True)



    def _get_reference_results(self, subset: str) -> tuple:
        """ Get the reference model's predictions and accuracy for the validation set.
        Args:
            subset (str): The subset to evaluate on (e.g., "train", "val" or "test").
        Returns:
            tuple: (ref_model_preds, accuracy, log_loss) of the reference model.
        """
        ref_model = ReferenceModel(self.FiLM_model, self.dataloader, self.z_dim)
        ref_model_preds, ref_acc = ref_model.get_probabilties_and_accuracy(subset)
        y_true = getattr(self.dataloader, f'y_{subset}')
        ref_loss = log_loss(y_true, ref_model_preds)
        return ref_model_preds, ref_acc, ref_loss
    

    def _get_z_candidates_from_df(self, df):
        z_columns = [f'z_{i}' for i in range(self.z_dim)]
        z_values = df[z_columns].to_numpy()
        return z_values
    

    def _evaluate_one_model(self, z_candidate: np.ndarray) -> tuple:
        """
        Evaluate a single model from the Pareto front.
        Args:
            z_candidate (np.ndarray): The z vector of the candidate model to evaluate.
        Returns:
            tuple: (preds, accuracy, log_loss, in_rashomon_set) of the model on the test set.
        """
        cand_preds = self.FiLM_model.predict(self.dataloader.x_test, z_candidate)
        cand_loss = log_loss(self.dataloader.y_test, cand_preds)
        cand_acc = np.mean(np.argmax(cand_preds, axis=1) == self.dataloader.y_test)
        in_rashomon_set = rashomon_check(cand_loss, self.ref_loss, self.epsilon)
        return cand_preds, cand_acc, cand_loss, in_rashomon_set

    def load_experiment_data(self):
        """
        Load all experiment data from .npz files into a list of DataFrames.
        """
        dfs = []

        for sigma in self.sigmas:
            sigma_path = Path(self.experiment_folder).joinpath(sigma)
            if not sigma_path.exists():
                print(f"[WARNING] Sigma path {sigma_path} does not exist. Skipping.", flush=True)
                continue

            z_folders = os.listdir(sigma_path)

            for z_folder in z_folders:
                z_path = sigma_path.joinpath(z_folder)
                if not z_path.exists():
                    print(f"[WARNING] Z path {z_path} does not exist. Skipping.", flush=True)
                    continue

                candidates_file = z_path.joinpath("candidates.npz")
                if not candidates_file.exists():
                    print(f"[WARNING] Candidates file {candidates_file} does not exist. Skipping.", flush=True)
                    continue

                data = np.load(candidates_file, allow_pickle=True)['data']
                df = pd.DataFrame(data)
                # Filter out rows that are not in the validation rashomon set
                df = df[df["is_in_rashomon_set"] == True].reset_index(drop=True)
                df['sigma'] = sigma
                df['z_folder'] = z_folder
                dfs.append(df)
        return dfs
        

    def evaluate_per_seed(self):
        for df in self.experiment_dfs:
            if df.shape[0] == 0:
                print(f"[INFO] Skipping empty DataFrame", flush=True)
                continue
            z_folder = df['z_folder'].iloc[0]
            sigma = df['sigma'].iloc[0]
            print(f"[INFO] Evaluating z_folder {z_folder}, sigma {sigma}, shape =", df.shape, flush=True)
            self.evaluate_df(df, sigma, z_folder, self.output_folder)


    def evaluate_df(self, df, sigma, z_folder, out_path):
        z_mat = self._get_z_candidates_from_df(df)
        total = len(z_mat)
        results = [None] * total  # Placeholder for all results
        for i, z_candidate in enumerate(z_mat):
            results[i] = self._evaluate_one_model(z_candidate)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Evaluated {i + 1}/{total} models in z_folder {z_folder}, sigma {sigma}", flush=True)

        # Unpack results
        preds_list, acc_list, loss_list, in_rashomon_list = zip(*results)
        rashomon_out = [(p, a, l, i) for (p, a, l, i) in results if i]
        rashomon_preds, rashomon_accs, rashomon_losses, _ = zip(*rashomon_out) if rashomon_out else ([], [], [], [])
        if len(rashomon_out) == 0:
            print(f"[WARNING] No models in the Rashomon set for z_folder {z_folder}, sigma {sigma}. Skipping Rashomon metrics.", flush=True)
            rashomon_metrics = pd.DataFrame({
            'z_dim': [self.z_dim],
            'epsilon': [self.epsilon],
            'sigma': [sigma],
            'z_folder': [z_folder],
            'test_rashomon_size': [len(rashomon_out)],
            'ambiguity': [None],
            'discrepancy': [None],
            'mean_vpr': [None],
            'rashomon_capacity': [None]
            })
            rashomon_metrics.to_csv(out_path / "rashomon_metrics" / f"rashomon_metrics_{z_folder}_{sigma}.csv", index=False)
            df.to_csv(out_path / "model_evaluations" / f"model_evaluations_{z_folder}_{sigma}.csv", index=False)
            return
        rashomon_probs = np.stack(rashomon_preds, axis=0)
        rashomon_labels = np.argmax(rashomon_probs, axis=2)
        amb, disc, vpr_width, rc_mean = self.get_rashomon_metrics(rashomon_probs, rashomon_labels)
        # Save results to DataFrame
        df['test_accuracy'] = acc_list
        df['test_log_loss'] = loss_list
        df['in_rashomon_set_test'] = in_rashomon_list
        # Rashomon metrics are in a separate csv file
        rashomon_metrics = pd.DataFrame({
            'z_dim': [self.z_dim],
            'epsilon': [self.epsilon],
            'sigma': [sigma],
            'z_folder': [z_folder],
            'test_rashomon_size': [len(rashomon_losses)],
            'ambiguity': [amb],
            'discrepancy': [disc],
            'mean_vpr': [vpr_width],
            'rashomon_capacity': [rc_mean]
        })
        rashomon_metrics.to_csv(out_path / "rashomon_metrics" / f"rashomon_metrics_{z_folder}_{sigma}.csv", index=False)
        df.to_csv(out_path / "model_evaluations" / f"model_evaluations_{z_folder}_{sigma}.csv", index=False)


    def get_rashomon_metrics(self, rashomon_probs, rashomon_labels):
        rashomon_scores = rashomon_probs
        rashomon_decisions = rashomon_labels
        # Probability-based
        score_y = true_class_probablities(rashomon_scores, self.dataloader.y_test)
        vpr_min, vpr_max, vpr_width = viable_prediction_range(score_y) # 0 is no viable prediction range, 1 is all predictions are viable
        # Decision-based 
        amb = ambiguity(rashomon_decisions, np.argmax(self.ref_preds, axis=1)) # scalar, that represents the % of test samples, where at least one model disagrees with the reference model
        disc = discrepancy(rashomon_decisions, np.argmax(self.ref_preds, axis=1)) # scalar, there is at least one model in the Rashomon set that disagrees from the base model on scalar% of the test set
        # score-based: Rashomon capacity (exact match)
        rc_per_sample = rashomon_capacity(rashomon_scores)  # (N,) bits
        rc_effective = 2 ** rc_per_sample  # (N,) effective number of models
        rc_mean = float(rc_effective.mean())
        return amb, disc, vpr_width.mean(), rc_mean


def handle_cli_args():
    parser = argparse.ArgumentParser(description="Evaluate CMA-ES candidates on a test set.")
    parser.add_argument("--model_type", type=str, choices=["mnist", "resnet50_pneumonia", "vgg16_cifar10"], required=True, help="Type of the model (e.g., 'mnist').")
    parser.add_argument("--epsilon", type=float, choices=[0.01, 0.02, 0.03, 0.04, 0.05], required=True, help="Epsilon value for the Rashomon set.")
    parser.add_argument("--z_dim", type=int, choices=[1, 2, 4, 8, 16, 32, 64], required=True, help="Dimension of the z vector.")
    return parser.parse_args()


if __name__ == "__main__": 
    # Handle CLI arguments
    args = handle_cli_args()
    cma = CMAEvaluator(seed=42, epsilon=args.epsilon, model_type=args.model_type, z_dim=args.z_dim, experiment_folder=f"cma_experiments/{args.model_type}/cma_{args.model_type}_epsilon_{args.epsilon}/z_{args.z_dim}")
    cma.evaluate_per_seed()
