import time
import argparse

from utils.experiment_parameters import get_experiment_parameters, get_cma_hyperparams
from utils.data_loader import Dataloader

from CMA.CMA_evolution_strategy import CMAEvolutionStrategy

def handle_cli_args():
    parser = argparse.ArgumentParser(description="CMA Timing Script")
    parser.add_argument("--z_dim", type=int, choices=[2, 4, 8, 16, 32, 64], help="Dimension of the latent vector z", required=True)
    parser.add_argument("--model_type", type=str, choices=["mnist", "resnet50_pneumonia", "vgg16_cifar10"], help="Type of model to use", required=True)
    parser.add_argument("--z_seed", type=str, choices=["z0_ones.npy", "z0_seed_0.npy", "z0_seed_1.npy", "z0_seed_2.npy", "z0_seed_3.npy",
                                                        "z0_seed_4.npy", "z0_seed_5.npy", "z0_seed_6.npy", "z0_seed_7.npy", "z0_zeros.npy"],
                                                          help="Which z_0 seed to use", required=True) # This doesn't matter for timing
    parser.add_argument("--sigma", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5], help="Initial standard deviation for CMA-ES", required=True) # This also doesn't matter for timing
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_cli_args()
    params = get_experiment_parameters(args.model_type)
    popsize, generations = get_cma_hyperparams(args.z_dim)
    dataloader = Dataloader(params["train_x_path"], params["train_y_path"],
                            params["val_x_path"], params["val_y_path"],
                            params["test_x_path"], params["test_y_path"])
    cma_es = CMAEvolutionStrategy(reference_model=params["reference_model"],
                                  popsize=popsize,
                                  dataloader=dataloader,
                                  model_type=params["model_type"],
                                  exp_name=f"timing_{args.model_type}_z{args.z_dim}_sigma{args.sigma}",
                                  z_dim=args.z_dim,
                                  input_shape=params["input_shape"],
                                  z_0_file=f"z_seeds/z_{args.z_dim}/{args.z_seed}",
                                  sigma_0=args.sigma,
                                  seed=42,
                                  epsilon=0.05)  # Epsilon can be any value here, as we are only timing
    start_time = time.time()
    # Your main code logic here
    cma_es.run(generations=generations)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")