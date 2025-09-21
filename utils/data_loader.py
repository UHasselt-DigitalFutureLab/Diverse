import numpy as np

class Dataloader:
    def __init__(self, test_x_path, test_y_path, train_x_path=None, train_y_path=None, val_x_path=None, val_y_path=None):
        self.x_test, self.y_test = self._load_data(test_x_path, test_y_path)
        if train_x_path and train_y_path:
            self.x_train, self.y_train = self._load_data(train_x_path, train_y_path)
        if val_x_path and val_y_path:
            self.x_val, self.y_val = self._load_data(val_x_path, val_y_path)
        

    def _handle_y_oh_or_not(self, y):
        """
        Handle the case where y is one-hot encoded or not.
        If y is one-hot encoded, convert it to integer labels.
        If y is not one-hot encoded, return it as is.
        Args:
            y (np.ndarray): The labels, either one-hot encoded or integer labels.
        Returns:
            np.ndarray: The labels as integer labels.
        """
        if y.ndim == 2 and y.shape[1] > 1:
            # If y_test is one-hot encoded, convert it to integer labels
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.squeeze()  # Ensure y is a 1D array
        return y_true
    

    def _load_data(self, x_path, y_path):
        """
        Load data from the specified paths.
        Test data is automatically set to a 1D vector to make sure it is the same if one-hot encoded or not
        Args:
            x_path (str): Path to the input data (features).
            y_path (str): Path to the output data (labels).
        Returns:
            Tuple[np.ndarray, np.ndarray]: Loaded input features and labels.
        """
        x_train = np.load(x_path)
        y_train = self._handle_y_oh_or_not(np.load(y_path))
        return x_train, y_train