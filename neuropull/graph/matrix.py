from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, lil_array

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]


class _BaseMatrix:
    def __init__(self, matrix) -> None:
        self._matrix = matrix

    @abstractmethod
    def reindex(self, index: Index, columns: Index) -> "_BaseMatrix":
        pass

    @property
    @abstractmethod
    def index(self):
        pass

    @property
    @abstractmethod
    def columns(self):
        pass

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if matrix.index.equals(self.index) and matrix.columns.equals(self.columns):
            self._matrix = matrix
        else:
            ValueError("New matrix index and columns must match the index and columns")

    def __repr__(self):
        return self._matrix.__repr__()


class DenseMatrix(_BaseMatrix):
    def reindex(self, index: Index, columns: Index) -> "DenseMatrix":
        matrix = self._matrix.reindex(index=index, columns=columns, fill_value=0)
        self._matrix = matrix
        return self

    @property
    def index(self):
        return self._matrix.index

    @property
    def columns(self):
        return self._matrix.columns


class SparseMatrix(_BaseMatrix):
    def __init__(self, matrix: csr_array, index: Index, columns: Index) -> None:
        super().__init__(matrix)

        if matrix.shape[0] != len(index):
            raise ValueError("Index length does not match matrix shape")
        if matrix.shape[1] != len(columns):
            raise ValueError("Columns length does not match matrix shape")

        self._reset_indexing_maps(index, columns)

    def _reset_indexing_maps(self, index, columns):
        self._index_map = pd.Series(index=index, data=np.arange(self._matrix.shape[0]))
        self._columns_map = pd.Series(
            index=columns, data=np.arange(self._matrix.shape[1])
        )

    def reindex(self, index: Index, columns: Index) -> "SparseMatrix":
        # TODO I think the smarter thing to do here is never instantiate the matrix
        # until the user asks for it. Basically just keep an edgelist in the meantime.
        # Then just remap indices when doing the reindex operations. But alas.
        new_index_map = self._index_map.reindex(index).astype("Int64")
        new_columns_map = self._columns_map.reindex(columns).astype("Int64")
        new_matrix = lil_array((len(new_index_map), (len(new_columns_map))))
        row_positions = np.arange(len(new_index_map))
        col_positions = np.arange(len(new_columns_map))
        # these are the indices in the old array, for all objects which were there
        valid_new_index_map = new_index_map[new_index_map.notna()]
        valid_new_row_positions = row_positions[new_index_map.notna()]
        valid_new_columns_map = new_columns_map[new_columns_map.notna()]
        valid_new_col_positions = col_positions[new_columns_map.notna()]
        new_matrix[
            np.ix_(valid_new_row_positions, valid_new_col_positions)
        ] = self._matrix[valid_new_index_map][:, valid_new_columns_map]
        new_matrix = csr_array(new_matrix)
        self._matrix = new_matrix
        self._reset_indexing_maps(index, columns)
        return self

    @property
    def index(self):
        return self._index_map.index

    @property
    def columns(self):
        return self._columns_map.index


class MultiMatrix:
    def __init__(self, matrices, layers=None) -> None:
        if layers is None:
            layers = pd.Index(np.arange(len(matrices)))
        if len(layers) != len(matrices):
            raise ValueError("Number of layers must match number of matrices")

        # TODO check that all matrices have the same shape and index/columns
        # TODO store matrices in a dictionary? or list?
        self._matrices = matrices

    @property
    def layers(self):
        return self._layers

    def reindex_layers(self, layers):
        raise NotImplementedError()

    def reindex(self, index: Index, columns: Index) -> "MultiMatrix":
        for matrix in self._matrices:
            matrix.reindex(index, columns)
        return self


# class MultiDenseMatrix(_BaseMultiMatrix):
#     def __init__(self, matrices) -> None:
#         super().__init__(matrices)


# class MultiSparseMatrix(_BaseMultiMatrix):
#     def __init__(self, matrices) -> None:
#         super().__init__(matrices)
class BaseFrame:
    def __init__(self, frame) -> None:
        pass


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

        if not row_objects.index.equals(matrix.index):
            raise ValueError("Row objects index does not match matrix index")
        if not col_objects.index.equals(matrix.columns):
            raise ValueError("Column objects index does not match matrix columns")

        self._matrix = matrix
        self._row_objects = row_objects
        self._col_objects = col_objects

    def reindex(self, index: Index, columns: Index) -> "MatrixFrame":
        self._matrix = self._matrix.reindex(index=index, columns=columns)
        self._row_objects = self._row_objects.reindex(index=index)
        self._col_objects = self._col_objects.reindex(index=columns)
        return self

    @property
    def index(self):
        return self._matrix.index

    @property
    def columns(self):
        return self._matrix.columns

    @property
    def matrix(self):
        return self._matrix.matrix

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

    @property
    def row_objects(self):
        return self._row_objects

    @row_objects.setter
    def row_objects(self, row_objects: pd.DataFrame):
        if not row_objects.index.equals(self.index):
            raise ValueError("Row objects index does not match matrix index")
        self._row_objects = row_objects

    @property
    def col_objects(self):
        return self._col_objects

    @col_objects.setter
    def col_objects(self, col_objects: pd.DataFrame):
        if not col_objects.index.equals(self.columns):
            raise ValueError("Column objects index does not match matrix columns")
        self._col_objects = col_objects

    def __repr__(self):
        return self._matrix.__repr__()


class MultiMatrixFrame(BaseFrame):
    def __init__(self):
        pass