import math

def get_experiment_parameters(model_type):
    """
    Get the parameters for the experiment based on the experiment name.
    This function is used to set the parameters for the CMA-ES algorithm.
    Args:
        model_type (str): The type of model/experiment. Options are "mnist",
                          "resnet50_pneumonia", "vgg16_cifar10", "vision_transformer_cifar10".
    Returns:
        dict: A dictionary containing the experiment parameters.
    *  ``reference_model``: filename of the pretrained reference model.
    *  ``model_type``     : type of model / experiment.
    *  ``input_shape``    : shape of the input data (height, width, channels).
    *  ``max_workers``    : number of parallel workers for running experiments.
    *  ``train_x_path``   : path to the training input data.
    *  ``train_y_path``   : path to the training output data.
    *  ``val_x_path``     : path to the validation input data.
    *  ``val_y_path``     : path to the validation output data.
    *  ``test_x_path``    : path to the test input data.
    *  ``test_y_path``    : path to the test output data.

    """
    if model_type == "mnist":
        return {
            "reference_model": "mnist_base.keras",
            "model_type": "mnist",
            "input_shape": (28, 28),
            "max_workers": 12, # Number of parallel workers for running experiments, should be set based on hardware capabilities
            "train_x_path": "datasets/mnist/x_train.npy",
            "train_y_path": "datasets/mnist/y_train.npy",
            "val_x_path": "datasets/mnist/x_val.npy",
            "val_y_path": "datasets/mnist/y_val.npy",
            "test_x_path": "datasets/mnist/x_test.npy",
            "test_y_path": "datasets/mnist/y_test.npy"
        }
    elif model_type == "resnet50_pneumonia":
        return {
            "reference_model": "resnet50_pneumonia.keras",
            "model_type": "resnet50_pneumonia",
            "input_shape": (224, 224, 3),
            "max_workers": 5, # Number of parallel workers for running experiments, should be set based on hardware capabilities
            "train_x_path": "datasets/pneumonia_mnist/x_train_normalized.npy",
            "train_y_path": "datasets/pneumonia_mnist/y_train_onehot.npy",
            "val_x_path": "datasets/pneumonia_mnist/x_val_normalized.npy",
            "val_y_path": "datasets/pneumonia_mnist/y_val_onehot.npy",
            "test_x_path": "datasets/pneumonia_mnist/x_test_normalized.npy",
            "test_y_path": "datasets/pneumonia_mnist/y_test_onehot.npy",
        }
    elif model_type == "vgg16_cifar10":
        return {
            "reference_model": "vgg16_cifar10.keras",
            "model_type": "vgg16_cifar10",
            "input_shape": (32, 32, 3),
            "max_workers": 4, # Number of parallel workers for running experiments, should be set based on hardware capabilities
            "train_x_path": "datasets/cifar10_vgg16/x_train.npy",
            "train_y_path": "datasets/cifar10_vgg16/y_train.npy",
            "val_x_path": "datasets/cifar10_vgg16/x_val.npy",
            "val_y_path": "datasets/cifar10_vgg16/y_val.npy",
            "test_x_path": "datasets/cifar10_vgg16/x_test.npy",
            "test_y_path": "datasets/cifar10_vgg16/y_test.npy",
        }
    elif model_type == "vision_transformer_cifar10":
            return {
                "reference_model": "vision_transformer.keras",
                "model_type": "vision_transformer_cifar10",
                "input_shape": (32, 32, 3),
                "max_workers": 4, # Number of parallel workers for running experiments, should be set based on hardware capabilities
                "train_x_path": "datasets/cifar10_vit/x_train.npy",
                "train_y_path": "datasets/cifar10_vit/y_train.npy",
                "val_x_path": "datasets/cifar10_vit/x_val.npy",
                "val_y_path": "datasets/cifar10_vit/y_val.npy",
                "test_x_path": "datasets/cifar10_vit/x_test.npy",
                "test_y_path": "datasets/cifar10_vit/y_test.npy",
            }
    else:
        raise ValueError(f"Unknown experiment name: {model_type}")


def get_cma_hyperparams(z_dim):
    """ Get the CMA-ES hyperparameters based on the latent vector dimension.
    Args:
        z_dim (int): Dimension of the latent vector.
    Returns:
        tuple: A tuple containing (popsize, generations).
    *  ``popsize``    : population size for CMA-ES.
    *  ``generations``: number of generations to run CMA-ES.
    """
    # The default CMA-ES setting (Hansen 2016), https://arxiv.org/pdf/1604.00772
    popsize = 4 + int(3 * math.log(z_dim))
    # k is the number of evaluations per candidate, should be between 60 and 100,
    # which is done often in practice and following the CMA-ES javadoc
    k = 80
    generations = math.ceil(k * z_dim / popsize)
    return popsize, generations
