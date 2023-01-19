"""Matrix frames."""

from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from .types import Index


class _BaseMatrix:
    def __init__(self, matrix) -> None:
        self._matrix = matrix

    @abstractmethod
    def reindex(
        self,
        index: Optional[Index] = None,
        columns: Optional[Index] = None,
        inplace=False,
    ) -> "_BaseMatrix":
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

    @property
    def data(self):
        return self.matrix

    def __repr__(self):
        return self._matrix.__repr__()


class DenseMatrix(_BaseMatrix):
    """Class for representing a dense matrix with row and column metadata."""

    def __init__(
        self, matrix: Union[pd.DataFrame, np.ndarray], index=None, columns=None
    ) -> None:
        if isinstance(matrix, np.ndarray):
            matrix = pd.DataFrame(matrix, index=index, columns=columns)
        self._matrix = matrix

    def reindex(
        self,
        index: Optional[Index] = None,
        columns: Optional[Index] = None,
        inplace=False,
    ) -> "DenseMatrix":
        """Reindex the matrix and row and column metadata.

        Parameters
        ----------
        index : Optional[Index], optional
            _description_, by default None
        columns : Optional[Index], optional
            _description_, by default None
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        DenseMatrix
            _description_
        """
        matrix = self._matrix.reindex(index=index, columns=columns, fill_value=0)
        if not inplace:
            return DenseMatrix(matrix)
        else:
            self._matrix = matrix
            return self

    @property
    def index(self):
        """Row index of the matrix and metadata."""
        return self._matrix.index

    @property
    def columns(self):
        """Column index of the matrix and metadata."""
        return self._matrix.columns

    @property
    def data(self):
        """Matrix data."""
        return self.matrix.values


class SparseMatrix(_BaseMatrix):
    """Class for representing a sparse matrix with row and column metadata."""

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

    def reindex(
        self,
        index: Optional[Index] = None,
        columns: Optional[Index] = None,
        inplace=False,
    ) -> "SparseMatrix":
        """Reindex the matrix and row and column metadata."""
        if inplace:
            raise NotImplementedError(
                "Inplace reindexing not implemented for sparse matrix"
            )

        if index is None:
            index = self.index
        if columns is None:
            columns = self.columns

        # this is a map from new index (in new index sorted order) to old index
        reordered_index_map = self._index_map.reindex(index).astype("Int64")
        reordered_columns_map = self._columns_map.reindex(columns).astype("Int64")
        # so this is a map from old positional index to new positional index
        old_to_new_index_pos_map = pd.Series(
            index=reordered_index_map.values, data=np.arange(len(reordered_index_map))
        )
        old_to_new_columns_pos_map = pd.Series(
            index=reordered_columns_map.values,
            data=np.arange(len(reordered_columns_map)),
        )

        # TODO: not sure how slow this is but probably could avoid getting all nonzeros
        old_edge_row_indices, old_edge_col_indices = np.nonzero(self.matrix)

        # mask to only keep edges that are in the new index
        row_in_new_mask = np.isin(old_edge_row_indices, old_to_new_index_pos_map.index)
        col_in_new_mask = np.isin(
            old_edge_col_indices, old_to_new_columns_pos_map.index
        )
        in_new_mask = row_in_new_mask & col_in_new_mask
        old_edge_row_indices = old_edge_row_indices[in_new_mask]
        old_edge_col_indices = old_edge_col_indices[in_new_mask]

        # for the edges that were there, get the new index position
        new_edge_row_indices = old_to_new_index_pos_map.loc[old_edge_row_indices].values
        new_edge_col_indices = old_to_new_columns_pos_map.loc[
            old_edge_col_indices
        ].values

        # get the old edge data
        edge_data = self.matrix[old_edge_row_indices, old_edge_col_indices]

        # put in place in the new matrix
        new_matrix = csr_array(
            (edge_data, (new_edge_row_indices, new_edge_col_indices)),
            shape=(len(index), len(columns)),
        )
        out = SparseMatrix(new_matrix, index, columns)
        return out

    @property
    def index(self):
        """Row index of the matrix and metadata."""
        return self._index_map.index

    @property
    def columns(self):
        """Column index of the matrix and metadata."""
        return self._columns_map.index


class MultiMatrix:
    """Class for representing a matrix with multiple layers."""

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
        """Layer names."""
        return self._layers

    def reindex_layers(self, layers):
        """Reindex the layers."""
        raise NotImplementedError()

    def reindex(self, index: Index, columns: Index) -> "MultiMatrix":
        """Reindex the matrix and row and column metadata."""
        for matrix in self._matrices:
            matrix.reindex(index, columns)
        return self


# class MultiDenseMatrix(_BaseMultiMatrix):
#     def __init__(self, matrices) -> None:
#         super().__init__(matrices)


# class MultiSparseMatrix(_BaseMultiMatrix):
#     def __init__(self, matrices) -> None:
#         super().__init__(matrices)
