from collections import defaultdict
from typing import Dict

import pandas as pd


class ErrorState:

    def __init__(self, index: pd.Index) -> None:
        self.errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.masks = defaultdict(lambda: pd.Series(False, index=index))

    def add_errors(self, mask: pd.DataFrame, col: str, details: Dict[str, str]) -> None:
        for index in mask[mask].index:
            self.errors[index][col]['details'].append(details)
        self.masks[col] = self.masks[col] | mask
