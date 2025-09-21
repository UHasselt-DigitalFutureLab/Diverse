import numpy as np
from utils.data_loader import Dataloader
from FiLM.FiLMModel import FiLMModel


class ReferenceModel:
    def __init__(self, film_model: FiLMModel, dataloader: Dataloader, z_dim: int):
        self.film_model = film_model
        self.dataloader = dataloader
        self.z_ref = np.zeros(z_dim)


    def get_probabilties_and_accuracy(self, subset: str):
        """Get the predicted probabilities and accuracy for a given subset.
        Args:
            subset (str): "train", "val" or "test"
        Returns:
            Tuple[np.ndarray, float]: predicted probabilities and accuracy
        """
        if subset == "train":
            ref_probs = self.film_model.predict(x=self.dataloader.x_train, z=self.z_ref)
            y_true = self.dataloader.y_train
        elif subset == "test":
            ref_probs = self.film_model.predict(x=self.dataloader.x_test, z=self.z_ref)
            y_true = self.dataloader.y_test
        elif subset == "val":
            ref_probs = self.film_model.predict(x=self.dataloader.x_val, z=self.z_ref)
            y_true = self.dataloader.y_val
        else:
            raise ValueError("Subset must be 'train' or 'validation'.")
        y_pred = np.argmax(ref_probs, axis=1)
        return ref_probs, np.mean(y_pred == y_true)