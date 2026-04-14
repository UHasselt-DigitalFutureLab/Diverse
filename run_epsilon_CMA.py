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
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to use", choices=["mnist", "resnet50_pneumonia", "vgg16_cifar10", "vision_transformer_cifar10"])
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for the Rashomon requirement", choices=[0.01, 0.02, 0.03, 0.04, 0.05])
    parser.add_argument("--lambda_val", type=float, required=False, default=0.5, help="Lambda value used as a mixing weight for hard and soft disagreement", choices=[0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_cli_args()
    model_type = args.model_type
    print(model_type)
    # Initialize the CMASearch object with the experiment name
    exp_params = get_experiment_parameters(model_type)
    cma_search = CMASearch(model_type=args.model_type, max_workers=exp_params["max_workers"], epsilon=args.epsilon, lambda_val=args.lambda_val)
    cma_search.run_all_experiments()
