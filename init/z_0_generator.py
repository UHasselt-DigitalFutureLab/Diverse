import numpy as np
import argparse
from pathlib import Path

def handle_cli_args():
    parser = argparse.ArgumentParser(description="Generate z_0 vectors for testing.")
    parser.add_argument("--z_dim", type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128],
                         required=True, help="Dimension of the z vector.")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    args = handle_cli_args()
    z_dim = args.z_dim
    # Script is run from the root directory
    z_path = Path(f"z_seeds/z_{z_dim}")
    z_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    NUM_VECTORS = 10 # We also include only ones and only zeros, so we need 8 more



    for i in range(NUM_VECTORS - 2):
        if i % 2 == 0:
            # Generate a vector with values close to 1.0
            z_0 = np.random.normal(loc=1.0, scale=0.1, size=z_dim)
        else:
            # Generate a vector with values close to 0.0
            z_0 = np.random.normal(loc=0.0, scale=0.1, size=z_dim)

        np.save(z_path.joinpath(f"z0_seed_{i}.npy"), z_0)

    # Save the all-ones and all-zeros vectors
    np.save(z_path.joinpath("z0_ones.npy"), np.ones(z_dim))
    np.save(z_path.joinpath("z0_zeros.npy"), np.zeros(z_dim))
