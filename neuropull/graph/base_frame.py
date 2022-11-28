from typing import Any, Optional

import pandas as pd

from .types import Index


class FrameGroupBy:
    def __init__(self, frame, row_objects_groupby, col_objects_groupby):
        self._frame = frame
        self._row_objects_groupby = row_objects_groupby
        self._col_objects_groupby = col_objects_groupby

        if row_objects_groupby is None:
            self._axis = 1
        elif col_objects_groupby is None:
            self._axis = 0
        else:
            self._axis = 'both'

        if self.has_row_groups:
            self.row_group_names = list(row_objects_groupby.groups.keys())
        if self.has_col_groups:
            self.col_group_names = list(col_objects_groupby.groups.keys())

    @property
    def has_row_groups(self):
        return self._row_objects_groupby is not None

    @property
    def has_col_groups(self):
        return self._col_objects_groupby is not None

    def __iter__(self):
        if self._axis == 'both':
            for row_group, row_objects in self._row_objects_groupby:
                for col_group, col_objects in self._col_objects_groupby:
                    yield (row_group, col_group), self._frame.reindex(
                        row_objects.index, col_objects.index
                    )
        elif self._axis == 0:
            for row_group, row_objects in self._row_objects_groupby:
                yield row_group, self._frame.reindex(row_objects.index)
        elif self._axis == 1:
            for col_group, col_objects in self._col_objects_groupby:
                yield col_group, self._frame.reindex(columns=col_objects.index)

    def apply(self, func, *args, data=False, **kwargs):
        if self._axis == 'both':
            answer = pd.DataFrame(
                index=self.row_group_names, columns=self.col_group_names
            )

        else:
            if self._axis == 0:
                answer = pd.Series(index=self.row_group_names)
            else:
                answer = pd.Series(index=self.col_group_names)
        for group, frame in self:
            if data:
                value = func(frame.data, *args, **kwargs)
            else:
                value = func(frame, *args, **kwargs)
            answer.at[group] = value
        return answer

    @property
    def row_groups(self):
        if self._axis == 'both' or self._axis == 0:
            return self._row_objects_groupby.groups
        else:
            raise ValueError('No row groups, groupby was on columns only')

    @property
    def col_groups(self):
        if self._axis == 'both' or self._axis == 1:
            return self._col_objects_groupby.groups
        else:
            raise ValueError('No col groups, groupby was on rows only')


class BaseFrame:
    def __init__(
        self, data: Any, row_objects: pd.DataFrame, col_objects: pd.DataFrame
    ) -> None:

        if not row_objects.index.equals(data.index):
            raise ValueError("Row objects index does not match matrix index")
        if not col_objects.index.equals(data.columns):
            raise ValueError("Column objects index does not match matrix columns")

        self._data = data
        self._row_objects = row_objects
        self._col_objects = col_objects

    def reindex(
        self,
        index: Optional[Index] = None,
        columns: Optional[Index] = None,
        inplace=False,
    ) -> "BaseFrame":
        data = self._data.reindex(index=index, columns=columns)
        row_objects = self._row_objects.reindex(index=index)
        col_objects = self._col_objects.reindex(index=columns)
        if not inplace:
            return self.__class__(data, row_objects, col_objects)
        else:
            self._data = data
            self._row_objects = row_objects
            self._col_objects = col_objects
            return self

    @property
    def index(self):
        return self._data.index

    @property
    def columns(self):
        return self._data.columns

    @property
    def data(self):
        return self._data.data

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

    def reindex_like(self, other: "BaseFrame") -> "BaseFrame":
        return self.reindex(index=other.index, columns=other.columns)

    def query(self, query: str, axis='both') -> "BaseFrame":
        row_objects = self.row_objects
        col_objects = self.col_objects
        if axis == 0:
            row_objects = row_objects.query(query)
        elif axis == 1:
            col_objects = col_objects.query(query)
        elif axis == 'both':
            row_objects = row_objects.query(query)
            col_objects = col_objects.query(query)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")
        return self.reindex(row_objects.index, col_objects.index)

    def groupby(self, by=None, axis='both', **kwargs):
        if axis == 0:
            row_objects_groupby = self.row_objects.groupby(by=by, **kwargs)
        elif axis == 1:
            col_objects_groupby = self.col_objects.groupby(by=by, **kwargs)
        elif axis == 'both':
            row_objects_groupby = self.row_objects.groupby(by=by, **kwargs)
            col_objects_groupby = self.col_objects.groupby(by=by, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")

        return FrameGroupBy(self, row_objects_groupby, col_objects_groupby)

    def sort_values(self, by, axis='both', **kwargs):
        row_objects = self.row_objects
        col_objects = self.col_objects
        if axis == 0:
            row_objects = row_objects.sort_values(by=by, inplace=False, **kwargs)
        elif axis == 1:
            col_objects = col_objects.sort_values(by=by, inplace=False, **kwargs)
        elif axis == 'both':
            row_objects = row_objects.sort_values(by=by, inplace=False, **kwargs)
            col_objects = col_objects.sort_values(by=by, inplace=False, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")
        return self.reindex(row_objects.index, col_objects.index)

    def set_index(self, keys, **kwargs):
        row_objects = self.row_objects.set_index(
            keys, verify_integrity=True, inplace=False, **kwargs
        )
        col_objects = self.col_objects.set_index(
            keys, verify_integrity=True, inplace=False, **kwargs
        )
        data = self._data
        new_data = self._data.__class__(
            data.data, index=row_objects.index, columns=col_objects.index
        )

        return self.__class__(new_data, row_objects, col_objects)
