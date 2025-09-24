import os
import json
import gc
import random
import time
import argparse
import multiprocessing as mp
from typing import Dict, Any, List
import numpy as np
# IMPORTANT: set threading env vars BEFORE any TensorFlow is imported in workers
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
from baselines.retraining_helper import get_mnist_data, train_and_evaluate_one_mnist_model, get_vgg_data, train_and_evaluate_one_vgg_model, get_resnet_data, train_and_evaluate_one_resnet_model

# Todo system for fallback if the system fails 
def _generate_combinations(start_seed, end_seed):
    """
    Generate all combinations of z_files and sigma_values.
    This function is used to create the tasks for the ProcessPoolExecutor.
    """
    return [{"seed": seed} for seed in range(start_seed, end_seed)]


def _load_or_create_todo_file(todo_file_path, start_seed, end_seed):
    """ Load the todo list from a file or create it if it doesn't exist."""
    if os.path.exists(todo_file_path):
        with open(todo_file_path, "r") as f:
            todo = json.load(f)
        print(f"[INFO] Loaded {len(todo)} tasks from {todo_file_path}")
    else:
        todo = _generate_combinations(start_seed, end_seed)
        with open(todo_file_path, "w") as f:
            json.dump(todo, f, indent=2)
        print(f"[INFO] Created {todo_file_path} with {len(todo)} tasks")
    return todo


def _save_todo_file(todo_file_path, todo):
    with open(todo_file_path, "w") as f:
        json.dump(todo, f, indent=2)



def _empty_buffer() -> Dict[str, List[Any]]:
    return {
        'seeds': [],
        'val_loss': [],
        'test_loss': [],
        'train_time': [],
        'test_preds': [],   # each item: (n_test,)
        'test_probs': []    # each item: (n_test, n_classes)
    }

def _save_chunk_npz(out_dir: str, model: str, seed: int, buffer: Dict[str, List[Any]]):
    """Save a buffer to an .npz chunk. Stacks along axis 0 (models)."""
    if len(buffer['seeds']) == 0:
        return

    path = os.path.join(out_dir, f"{model}_sweep_chunk_{seed:03d}.npz")

    # Convert lists to arrays; preds/probs should stack cleanly if test set size is consistent
    seeds      = np.asarray(buffer['seeds'], dtype=np.int64)
    val_loss   = np.asarray(buffer['val_loss'], dtype=np.float32)
    test_loss  = np.asarray(buffer['test_loss'], dtype=np.float32)
    train_time = np.asarray(buffer['train_time'], dtype=np.float32)

    # Stack predictions/probabilities
    # If shapes mismatch (e.g., a failed run appended different length), fall back to object arrays.
    def _safe_stack(lst):
        try:
            return np.stack(lst, axis=0)
        except Exception:
            arr = np.empty(len(lst), dtype=object)
            for i, x in enumerate(lst): arr[i] = np.asarray(x)
            return arr

    test_preds = _safe_stack(buffer['test_preds'])
    test_probs = _safe_stack(buffer['test_probs'])

    np.savez_compressed(
        path,
        seeds=seeds,
        val_loss=val_loss,
        test_loss=test_loss,
        train_time=train_time,
        test_preds=test_preds,
        test_probs=test_probs,
    )
    print(f"Saved {len(seeds)} models to: {path}", flush=True)


# ----------------------- worker -----------------------

def _train_one_resnet_seed(seed_value: int, out_dir: str) -> bool:
    """
    Trains one RESNET50 model with the given seed in a fresh process.
    Returns True if successful, False otherwise.
    """
    # Delay TensorFlow import to ensure process-level isolation and env vars take effect
    import tensorflow as tf

    # Extra safety: restrict TF threading inside the worker
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    # Optional: GPU memory growth (safe no-op on CPU-only)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    buffer = _empty_buffer()
    y_probs = y_preds = None

    try:
        print(f"[WORKER] Training seed {seed_value}", flush=True)

        # Reproducibility in this process
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        x_train, y_train, x_val, y_val, x_test, y_test = get_resnet_data(seed_value)

        start_time = time.perf_counter()
        (val_loss,
         test_loss,
         y_probs,
         y_preds,
         y_true) = train_and_evaluate_one_resnet_model(
            seed_value, x_train, y_train, x_val, y_val, x_test, y_test
        )
        end_time = time.perf_counter()

        buffer['seeds'].append(seed_value)
        buffer['val_loss'].append(val_loss)
        buffer['test_loss'].append(test_loss)
        buffer['train_time'].append(end_time - start_time)
        buffer['test_preds'].append(np.asarray(y_preds))
        buffer['test_probs'].append(np.asarray(y_probs))

        os.makedirs(out_dir, exist_ok=True)
        _save_chunk_npz(out_dir, "resnet", seed=seed_value, buffer=buffer)

        return True

    except Exception as e:
        print(f"[ERROR][seed={seed_value}] {e}", flush=True)
        return False

    finally:
        # Aggressive cleanup in the worker
        try:
            del x_train, y_train, x_val, y_val, x_test, y_test  # noqa: F821
        except Exception:
            pass
        try:
            del y_probs, y_preds, y_true  # noqa: F821
        except Exception:
            pass
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

def _train_one_vgg_seed(seed_value: int, out_dir: str) -> bool:
    """
    Trains one VGG model with the given seed in a fresh process.
    Returns True if successful, False otherwise.
    """
    # Delay TensorFlow import to ensure process-level isolation and env vars take effect
    import tensorflow as tf

    # Extra safety: restrict TF threading inside the worker
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    # Optional: GPU memory growth (safe no-op on CPU-only)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    buffer = _empty_buffer()
    y_probs = y_preds = None

    try:
        print(f"[WORKER] Training seed {seed_value}", flush=True)

        # Reproducibility in this process
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        x_train, y_train, x_val, y_val, x_test, y_test = get_vgg_data(seed_value)

        start_time = time.perf_counter()
        (val_loss,
         test_loss,
         y_probs,
         y_preds,
         y_true) = train_and_evaluate_one_vgg_model(
            seed_value, x_train, y_train, x_val, y_val, x_test, y_test
        )
        end_time = time.perf_counter()

        buffer['seeds'].append(seed_value)
        buffer['val_loss'].append(val_loss)
        buffer['test_loss'].append(test_loss)
        buffer['train_time'].append(end_time - start_time)
        buffer['test_preds'].append(np.asarray(y_preds))
        buffer['test_probs'].append(np.asarray(y_probs))

        os.makedirs(out_dir, exist_ok=True)
        _save_chunk_npz(out_dir, "vgg16", seed=seed_value, buffer=buffer)

        return True

    except Exception as e:
        print(f"[ERROR][seed={seed_value}] {e}", flush=True)
        return False

    finally:
        # Aggressive cleanup in the worker
        try:
            del x_train, y_train, x_val, y_val, x_test, y_test  # noqa: F821
        except Exception:
            pass
        try:
            del y_probs, y_preds, y_true  # noqa: F821
        except Exception:
            pass
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

def _train_one_mnist_seed(seed_value: int, out_dir: str) -> bool:
    """
    Trains one MNIST model with the given seed in a fresh process.
    Returns True if successful, False otherwise.
    """
    # Delay TensorFlow import to ensure process-level isolation and env vars take effect
    import tensorflow as tf

    # Extra safety: restrict TF threading inside the worker
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    # Optional: GPU memory growth (safe no-op on CPU-only)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    buffer = _empty_buffer()
    y_probs = y_preds = None

    try:
        print(f"[WORKER] Training seed {seed_value}", flush=True)

        # Reproducibility in this process
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data(seed_value)

        start_time = time.perf_counter()
        (val_loss,
         test_loss,
         y_probs,
         y_preds,
         y_true) = train_and_evaluate_one_mnist_model(
            seed_value, x_train, y_train, x_val, y_val, x_test, y_test
        )
        end_time = time.perf_counter()

        buffer['seeds'].append(seed_value)
        buffer['val_loss'].append(val_loss)
        buffer['test_loss'].append(test_loss)
        buffer['train_time'].append(end_time - start_time)
        buffer['test_preds'].append(np.asarray(y_preds))
        buffer['test_probs'].append(np.asarray(y_probs))

        os.makedirs(out_dir, exist_ok=True)
        _save_chunk_npz(out_dir, "mnist", seed=seed_value, buffer=buffer)

        return True

    except Exception as e:
        print(f"[ERROR][seed={seed_value}] {e}", flush=True)
        return False

    finally:
        # Aggressive cleanup in the worker
        try:
            del x_train, y_train, x_val, y_val, x_test, y_test  # noqa: F821
        except Exception:
            pass
        try:
            del y_probs, y_preds, y_true  # noqa: F821
        except Exception:
            pass
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def train_resnet_sweep(out_dir: str, todo: List[Dict[str, Any]], todo_file_path: str):
    """
    Parent resnet process orchestrator:
    - Spawns ONE worker process per seed (process is replaced after every seed).
    - No multithreading in the worker (TF threads set to 1).
    - Updates todo file only upon success.
    Args:
        out_dir (str): Directory to save output .npz files.
        todo (List[Dict[str, Any]]): List of tasks to perform, each with a 'seed' key.
        todo_file_path (str): Path to the todo JSON file for
    """
    print(out_dir, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    # Use 'spawn' to guarantee a clean interpreter for TF
    ctx = mp.get_context("spawn")

    # Pool with 1 process and maxtasksperchild=1 => new process for every task
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        # Iterate over a snapshot of todo to allow removal on success
        for task in list(todo):
            seed_value = int(task["seed"])
            print(f"[PARENT] Scheduling seed {seed_value}", flush=True)

            # Run synchronously (one at a time) but still in a fresh child proc
            result = pool.apply(_train_one_resnet_seed, (seed_value, out_dir))

            if result is True:
                # Remove from todo and persist
                todo.remove(task)
                _save_todo_file(todo_file_path, todo)
                print(f"[PARENT] Completed seed {seed_value}. Remaining: {len(todo)}", flush=True)
            else:
                print(f"[PARENT] Seed {seed_value} failed; leaving in todo for retry.", flush=True)

    print("[PARENT] Sweep finished (successful or attempted all tasks).", flush=True)

def train_vgg_sweep(out_dir: str, todo: List[Dict[str, Any]], todo_file_path: str):
    """
    Parent vgg process orchestrator:
    - Spawns ONE worker process per seed (process is replaced after every seed).
    - No multithreading in the worker (TF threads set to 1).
    - Updates todo file only upon success.
    Args:
        out_dir (str): Directory to save output .npz files.
        todo (List[Dict[str, Any]]): List of tasks to perform, each with a 'seed' key.
        todo_file_path (str): Path to the todo JSON file for
    """
    print(out_dir, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    # Use 'spawn' to guarantee a clean interpreter for TF
    ctx = mp.get_context("spawn")

    # Pool with 1 process and maxtasksperchild=1 => new process for every task
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        # Iterate over a snapshot of todo to allow removal on success
        for task in list(todo):
            seed_value = int(task["seed"])
            print(f"[PARENT] Scheduling seed {seed_value}", flush=True)

            # Run synchronously (one at a time) but still in a fresh child proc
            result = pool.apply(_train_one_vgg_seed, (seed_value, out_dir))

            if result is True:
                # Remove from todo and persist
                todo.remove(task)
                _save_todo_file(todo_file_path, todo)
                print(f"[PARENT] Completed seed {seed_value}. Remaining: {len(todo)}", flush=True)
            else:
                print(f"[PARENT] Seed {seed_value} failed; leaving in todo for retry.", flush=True)

    print("[PARENT] Sweep finished (successful or attempted all tasks).", flush=True)

def train_mnist_sweep(out_dir: str, todo: List[Dict[str, Any]], todo_file_path: str):
    """
    Parent mnist process orchestrator:
    - Spawns ONE worker process per seed (process is replaced after every seed).
    - No multithreading in the worker (TF threads set to 1).
    - Updates todo file only upon success.
    Args:
        out_dir (str): Directory to save output .npz files.
        todo (List[Dict[str, Any]]): List of tasks to perform, each with a 'seed' key.
        todo_file_path (str): Path to the todo JSON file for
    """
    print(out_dir, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    # Use 'spawn' to guarantee a clean interpreter for TF
    ctx = mp.get_context("spawn")

    # Pool with 1 process and maxtasksperchild=1 => new process for every task
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        # Iterate over a snapshot of todo to allow removal on success
        for task in list(todo):
            seed_value = int(task["seed"])
            print(f"[PARENT] Scheduling seed {seed_value}", flush=True)

            # Run synchronously (one at a time) but still in a fresh child proc
            result = pool.apply(_train_one_mnist_seed, (seed_value, out_dir))

            if result is True:
                # Remove from todo and persist
                todo.remove(task)
                _save_todo_file(todo_file_path, todo)
                print(f"[PARENT] Completed seed {seed_value}. Remaining: {len(todo)}", flush=True)
            else:
                print(f"[PARENT] Seed {seed_value} failed; leaving in todo for retry.", flush=True)

    print("[PARENT] Sweep finished (successful or attempted all tasks).", flush=True)

def handle_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models with different seeds.")
    parser.add_argument("--model", type=str, choices=["resnet", "vgg", "mnist"], required=True,
                        help="Type of model to train: 'resnet', 'vgg', or 'mnist'.")
    parser.add_argument("--start_seed", type=int, choices=[42, 45], required=True,
                        help="Starting seed value (inclusive).")
    parser.add_argument("--search_budget", type=int, choices=[162, 320, 640, 1284, 2562, 5120], required=True,
                        help="Number of models to train.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = handle_cli_args()
    start_seed = args.start_seed  # inclusive
    amount_to_train = args.search_budget
    end_seed = start_seed + amount_to_train  # exclusive

    if args.model == "resnet":
        todo_path = "todo/retraining_resnet.json"
        todo = _load_or_create_todo_file(todo_path, start_seed, end_seed)
        train_resnet_sweep(
            out_dir="baseline_evaluations/retraining/retraining_resnet",
            todo=todo,
            todo_file_path=todo_path,
        )
    elif args.model == "vgg":
        todo_path = "todo/retraining_vgg.json"
        todo = _load_or_create_todo_file(todo_path, start_seed, end_seed)
        train_vgg_sweep(
            out_dir="baseline_evaluations/retraining/retraining_vgg16",
            todo=todo,
            todo_file_path=todo_path,
        )
    elif args.model == "mnist":
        todo_path = "todo/retraining_mnist.json"
        todo = _load_or_create_todo_file(todo_path, start_seed, end_seed)
        train_mnist_sweep(
            out_dir="baseline_evaluations/retraining/retraining_mnist",
            todo=todo,
            todo_file_path=todo_path,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
