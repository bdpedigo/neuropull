from typing import Union

import numpy as np
import pandas as pd

from .base_frame import BaseFrame
from .matrix import DenseMatrix, SparseMatrix, csr_array


class MatrixFrame(BaseFrame):
    def __init__(
        self,
        matrix: Union[pd.DataFrame, np.ndarray, csr_array],
        row_objects: pd.DataFrame,
        col_objects: pd.DataFrame,
    ) -> None:

        if isinstance(matrix, np.ndarray):
            matrix = pd.DataFrame(matrix)

        if isinstance(matrix, pd.DataFrame):
            matrix = DenseMatrix(matrix)
        elif isinstance(matrix, csr_array):
            matrix = SparseMatrix(matrix, row_objects.index, col_objects.index)

        super().__init__(matrix, row_objects, col_objects)

    @property
    def matrix(self):
        return self._data.matrix

    @matrix.setter
    def matrix(self, matrix):
        # TODO not sure how to handle this for sparse arrays
        raise NotImplementedError("MatrixFrame.matrix is read-only")
        if not matrix.index.equals(self.index):
            raise ValueError("New matrix index does not match current index")
        if not matrix.columns.equals(self.columns):
            raise ValueError("New matrix columns does not match current columns")
        self._matrix.matrix = matrix

    @property
    def shape(self):
        return self._matrix.shape

    def __repr__(self):
        return self._matrix.__repr__()


class MultiMatrixFrame(BaseFrame):
    def __init__(self):
        pass
