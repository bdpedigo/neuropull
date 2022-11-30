"""Special types for the module."""

from typing import Union

import numpy as np
import pandas as pd

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]
