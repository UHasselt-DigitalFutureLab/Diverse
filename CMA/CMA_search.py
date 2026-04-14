import os
import json
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.data_loader import Dataloader
from utils.experiment_parameters import get_experiment_parameters, get_cma_hyperparams
from CMA.CMA_evolution_strategy import CMAEvolutionStrategy


def _run_cma_task(model_type: str, task: dict, epsilon: float, lambda_val: float):
    """
    Worker function executed in a separate process.
    Re-creates a CMASearch instance and runs ONE experiment.
    Args:
        model_type (str): Type of model / experiment.
        z0_seeds (str): Which set of z0 seeds to use ("old", "new", "negative").
        task (dict): A dictionary containing the parameters for the CMA-ES run.
        epsilon (float): Small constant for numerical stability in CMA-ES.
    Returns:
        dict: The input task dictionary, used to mark completion.
    """
    import os, tensorflow as tf
    # let TF grab memory dynamically instead of pre-allocating it all
    memory_limit = 1024 if model_type == "mnist" else  5120  # 1GB for MNIST, 5GB for others
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
        )

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    search = CMASearch(model_type=model_type, max_workers=1, epsilon=epsilon, lambda_val=lambda_val)
    search.run_one_cma_experiment(task["z_file"], task["sigma"], task["z_dim"])
    return task                       # used to mark completion


class CMASearch:
    def __init__(self, model_type, max_workers, epsilon, lambda_val):
        self.random_seed = 42
        self.model_type = model_type
        self.epsilon = epsilon
        self.max_workers = max_workers
        self.lambda_val = lambda_val
        self.experiment_name = f"cma_{model_type}_epsilon_{epsilon}_lambda_{lambda_val}"
        self.todo_file_path = Path(f"todo/{self.experiment_name}.json")
        os.makedirs(self.todo_file_path.parent, exist_ok=True)
        self.z_file_path = Path("z_seeds")
        self.sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.z_dims = [2, 4, 8, 16, 32, 64]
        self._todo = None


    # Lazy loading of the todo list, it only gets loaded when accessed
    # Thus, only when run_all
    @property
    def todo(self):
        if self._todo is None:
            self._todo = self._load_or_create_todo_file()
        return self._todo


    def _generate_combinations(self, z_files):
        """
        Generate all combinations of z_files and sigma_values.
        This function is used to create the tasks for the ProcessPoolExecutor.
        Args:
            z_files (list): List of z file names.
        Returns:
            list: A list of dictionaries, each containing a combination of z_file, sigma, and z_dim.
        *  ``z_file``: filename of the latent vector z.
        *  ``sigma`` : initial standard deviation for CMA-ES.
        *  ``z_dim`` : dimension of the latent vector z.
        """
        return [{"z_file": z, "sigma": s, "z_dim": z_dim} for z in z_files for s in self.sigma_values for z_dim in self.z_dims]


    def _load_or_create_todo_file(self):
        """ Load the todo list from a file or create it if it doesn't exist.
        Returns:
            list: A list of tasks to be executed.
        """
        if os.path.exists(self.todo_file_path):
            with open(self.todo_file_path, "r") as f:
                todo = json.load(f)
            print(f"[INFO] Loaded {len(todo)} tasks from {self.todo_file_path}")
        else:
            # We expect all z_files to be in the same format, for each z_dim
            z_files = os.listdir(self.z_file_path.joinpath(f"z_{self.z_dims[0]}"))
            todo = self._generate_combinations(z_files)
            with open(self.todo_file_path, "w") as f:
                json.dump(todo, f, indent=2)
            print(f"[INFO] Created {self.todo_file_path} with {len(todo)} tasks")
        return todo


    def _save_todo_file(self, todo):
        """ Save the current todo list to a file, used to overwrite the file after each completed task.
        Args:
            todo (list): The current todo list.
        """
        with open(self.todo_file_path, "w") as f:
            json.dump(todo, f, indent=2)


    def run_one_cma_experiment(self, z_0_file, sigma, z_dim):
        """ Run a single CMA-ES experiment with the given parameters.
        Args:
            z_0_file (str): Filename of the latent vector z.
            sigma (float): Initial standard deviation for CMA-ES.
            z_dim (int): Dimension of the latent vector z.
        """
        params = get_experiment_parameters(self.model_type)
        popsize, generations = get_cma_hyperparams(z_dim)
        dataloader = Dataloader(train_x_path=params["train_x_path"], train_y_path=params["train_y_path"],
                                val_x_path=params["val_x_path"], val_y_path=params["val_y_path"],
                                test_x_path=params["test_x_path"], test_y_path=params["test_y_path"])
        
        cma_es = CMAEvolutionStrategy(exp_name=self.experiment_name, reference_model=params["reference_model"],
                                        popsize=popsize, model_type=params["model_type"], z_dim=z_dim, input_shape=params["input_shape"],
                                        z_0_file=self.z_file_path.joinpath(f"z_{z_dim}/{z_0_file}"), sigma_0=sigma, seed=self.random_seed,
                                        dataloader=dataloader, epsilon=self.epsilon, lambda_val=self.lambda_val)

        cma_es.run(generations=generations)


    def run_all_experiments(self):
        """Run every item in self.todo in parallel, up to max_workers processes."""
        if not self.todo:
            print("[INFO] Nothing to do - all experiments already completed.")
            return

        # Shuffle so that long/short jobs are mixed for better load-balancing
        random.shuffle(self.todo)
        outstanding = {tuple(t.values()): t for t in self.todo}   # quick lookup
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_run_cma_task, self.model_type, task, self.epsilon, self.lambda_val): task
                for task in self.todo
            }

            for fut in as_completed(futures):
                task = futures[fut]
                key = tuple(task.values())
                try:
                    fut.result()           # raises if worker failed
                    outstanding.pop(key, None)
                except Exception as e:
                    print(f"[ERROR] {task} → {e}")

                # Persist remaining work after every finished job
                self._save_todo_file(list(outstanding.values()))