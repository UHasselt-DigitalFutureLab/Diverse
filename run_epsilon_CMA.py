""" 
    Script to run CMA-ES experiments with different z_0 vectors and sigma values but with one epsilon.
    This script uses multiprocessing to run multiple CMA-ES experiments in parallel.
    Running this will take a very long time, as all the separate CMA-ES experiments will be run.
 """

import argparse
from CMA.CMA_search import CMASearch
from utils.experiment_parameters import get_experiment_parameters

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def handle_cli_args():
    """ Handle command line arguments for the CMA-ES script. """
    parser = argparse.ArgumentParser(description="Run CMA-ES for different z_0 vectors and sigma values and different experiments.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to use", choices=["mnist", "resnet50_pneumonia", "vgg16_cifar10"])
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for the Rashomon requirement", choices=[0.01, 0.02, 0.03, 0.04, 0.05])
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_cli_args()
    model_type = args.model_type
    # Initialize the CMASearch object with the experiment name
    exp_params = get_experiment_parameters(model_type)
    cma_search = CMASearch(model_type=args.model_type, max_workers=exp_params["max_workers"], epsilon=args.epsilon)
    cma_search.run_all_experiments()
