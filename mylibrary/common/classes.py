import numpy as np

from mylibrary.common.utils import save_matrix

# class to store the DTW matrix
class DTW_Matrix:
    def __init__(self, qini, qend, dini, dend, data=None):
        self.qini = qini
        self.qend = qend
        self.dini = dini
        self.dend = dend
        self.data = data

    # check if the matrix is empty
    def is_empty(self):
        return self.data is None

    def create_empty_matrix(self):
        self.data = np.full((self.qend-self.qini, self.dend-self.dini), np.nan)

    def save_matrix(self, matrix_name):
        # Save the matrix safely
        save_matrix(matrix_name, self.data)
