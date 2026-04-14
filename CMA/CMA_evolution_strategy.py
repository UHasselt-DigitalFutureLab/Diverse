import os
import random
from pathlib import Path
import cma
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from FiLM.FiLMModel import FiLMModel
from utils.data_loader import Dataloader
from utils.reference_model import ReferenceModel
from utils.evaluation_metrics import rashomon_check, total_variation_distance


class CMAEvolutionStrategy:
    def __init__(self, reference_model : str, popsize : int, dataloader : Dataloader, model_type : str, exp_name : str, z_dim : int,
                  input_shape : tuple, z_0_file : str, sigma_0 : float, seed : int, epsilon : float, lambda_val : float):
        # Random seeds for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Loading the training and validation data
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.film_model = FiLMModel(reference_model, name=model_type, input_shape=input_shape, z_dim=self.z_dim, random_seed=seed)
        self.reference_model = ReferenceModel(self.film_model, self.dataloader, z_dim=self.z_dim)
        # Get the reference predictions and accuracy for both train and validation sets
        self.train_ref_preds, self.train_ref_acc = self.reference_model.get_probabilties_and_accuracy("train")
        self.train_ref_loss = log_loss(self.dataloader.y_train, self.train_ref_preds)
        self.val_ref_preds, self.val_ref_acc = self.reference_model.get_probabilties_and_accuracy("val")
        self.val_ref_loss = log_loss(self.dataloader.y_val, self.val_ref_preds)

        # Experiment specific parameters
        self.exp_name = exp_name
        self.lambda_val = lambda_val
        self.z_0 = np.load(z_0_file)
        self.sigma_0 = sigma_0
        self.model_type = model_type

        # Initialize the experiment with the specific parameters
        self.strategy = cma.CMAEvolutionStrategy(self.z_0, self.sigma_0, {'popsize': popsize, 'seed': seed})

        # Check if output directory exists, if not create it
        ## In that folder, the results will be stored in a pandas DataFrame
        z0_stem = Path(z_0_file).stem
        self.output_dir = (
            f"cma_experiments/{model_type}/{exp_name}/z_{self.z_dim}/sigma_{self.sigma_0}/{z0_stem}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    
    def gaussian_total_variation_fitness(self, z_candidate, lam=0.5):
        """
            Fitness function that combines diversity (measured by total variation distance and disagreement)
            with a Gaussian penalty based on the loss increase compared to the reference model.
            The parameter lam controls the trade-off between TVD and disagreement.
            The parameter epsilon controls the width of the Gaussian penalty.
            A smaller epsilon means a stricter penalty for loss increase.
            Args:
                z_candidate (np.ndarray): The candidate latent vector z.
                lam (float): The weight for combining TVD and disagreement. Default is 0.5.
            Returns:
                fitness (float): The fitness score of the candidate z.
                disagreement_mean (float): The mean hard disagreement with the reference model over the training dataset.
                acc (float): The accuracy of the candidate model on the training dataset.
                tvd_mean (float): The mean total variation distance with the reference model over the training dataset
                diversity (float): The combined diversity measure (lam * TVD + (1-lam) * disagreement)

        """
        # Let the FiLM model with candidate z make predictions
        cand_probs = self.film_model.predict(x=self.dataloader.x_train, z=z_candidate)
        cand_labels = np.argmax(cand_probs, axis=1)

        # Calculate the accuracy of the candidate z
        acc = accuracy_score(self.dataloader.y_train, cand_labels)

        # Cross-entropy loss between the candidate predictions and the true labels
        # This is used to penalize the candidate if it has a high loss compared to the
        loss = log_loss(self.dataloader.y_train, cand_probs)
        # Relative loss increase from reference model
        rel_loss_increase = (loss - self.train_ref_loss) / (self.train_ref_loss + 1e-8)
        # Gaussian penalty on relative loss increase (centered at 0)
        loss_penalty = np.exp(- (rel_loss_increase ** 2) / (2 * self.epsilon ** 2))
        # Guassian penalty for acccuracy, meaning it will try to keep the accuracy close to the reference model's accuracy
        #acc_penalty = np.exp(-((acc - self.train_ref_acc) ** 2) / (2 * epsilon ** 2))

        # Total variation distance, this will give a vector of shape (n_samples, )
        tvd = total_variation_distance(self.train_ref_preds, cand_probs)
        ## Calculate the mean TVD for the fitness
        tvd_mean = np.mean(tvd)

        # Hard disagreeement, this is the fraction of samples that have a different label than the reference model
        disagreement = cand_labels != np.argmax(self.train_ref_preds, axis=1)
        disagreement_mean = np.mean(disagreement)

        diversity = lam * tvd_mean + (1 - lam) * disagreement_mean

        fitness = diversity * loss_penalty

        return fitness, disagreement_mean, acc, tvd_mean, diversity
    

    def evaluate_z_on_val_set(self, z_candidate):
        """
            Evaluate the candidate z on the validation set and return the accuracy and the disagreement.
            This is used to monitor the performance of the candidate z on unseen data.
        """
        # Let the FiLM model with candidate z make predictions
        cand_probs = self.film_model.predict(x=self.dataloader.x_val, z=z_candidate)
        cand_labels = np.argmax(cand_probs, axis=1)
        acc = accuracy_score(self.dataloader.y_val, cand_labels)
        loss = log_loss(self.dataloader.y_val, cand_probs)

        # Soft disagreement
        tvd = total_variation_distance(self.val_ref_preds, cand_probs)
        tvd_mean = np.mean(tvd)

        # Hard disagreeement, this is the fraction of samples that have a different label than the reference model
        disagreement = cand_labels != np.argmax(self.val_ref_preds, axis=1)
        disagreement_mean = np.mean(disagreement)

        # Rashomon check: whether the candidate is in the Rashomon set
        is_in_rashomon_set = rashomon_check(cand_loss=loss, ref_loss=self.val_ref_loss, epsilon=self.epsilon)
        return cand_probs, acc, disagreement_mean, tvd_mean, loss, is_in_rashomon_set


    def run(self, generations):
        """
            Run the CMA-ES algorithm for a given number of generations.
            Each generation will generate a new candidate z and evaluate its fitness.
        """
        for gen_idx in range(generations):
            print(f"[{self.exp_name}] Generation {gen_idx+1}/{generations} | z_dim: {self.z_dim} | sigma: {self.sigma_0}", flush=True)
            try:
                # Perform one generation pass
                self._one_generation_pass(gen_idx)
            except Exception as e:
                print(f"[{self.exp_name}] Error in generation {gen_idx}: {e}", flush=True)
        # Combine the results of all generations into a single .npz file
        self._combine_generation_results(generations)


    def _combine_generation_results(self, generations):
        """
            Combine the results of all generations into a single .npz file.
            This will be used to save the results of the experiment.
        """
        dfs = []
        for gen_idx in range(generations):
            gen_path = os.path.join(self.output_dir, f"{self.exp_name}_gen_{gen_idx}.npz")
            data = np.load(gen_path)["data"]
            dfs.append(pd.DataFrame(data))

        df_full = pd.concat(dfs, ignore_index=True)

        final_path = os.path.join(self.output_dir, "candidates.npz")
        np.savez_compressed(final_path, data=df_full.to_records(index=False))

        # Clean up per-generation files
        for gen_idx in range(generations):
            os.remove(os.path.join(self.output_dir, f"{self.exp_name}_gen_{gen_idx}.npz"))


    def _one_generation_pass(self, gen_idx: int):
        """
            Perform one generation pass of the CMA-ES algorithm.
            This will generate a new candidate z and evaluate its fitness.
        """
        candidates = self.strategy.ask()
        records = []

        for i, z in enumerate(candidates):
            fitness, disagreement, acc, tvd, diversity = self.gaussian_total_variation_fitness(z, lam=self.lambda_val)
            _, val_acc, val_disagreement, val_tvd, val_loss, is_in_rashomon_set = self.evaluate_z_on_val_set(z)
            record = {
                "generation": gen_idx,
                "candidate_idx": i,
                "train_fitness": fitness,
                "train_accuracy": acc,
                "train_hard_disagreement": disagreement,
                "train_tvd": tvd,
                "train_diversity": diversity,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "val_hard_disagreement": val_disagreement,
                "val_tvd": val_tvd,
                "is_in_rashomon_set": is_in_rashomon_set,
                **{f"z_{j}": val for j, val in enumerate(z)}
            }
            records.append(record)

        self.strategy.tell(candidates, [-r["train_fitness"] for r in records])
        
        # Append to the experiment's log DataFrame and LOGGING
        # Save this generation only
        df_gen = pd.DataFrame(records)
        gen_path = os.path.join(self.output_dir, f"{self.exp_name}_gen_{gen_idx}.npz")
        np.savez_compressed(gen_path, data=df_gen.to_records(index=False))
