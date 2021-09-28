"""
Transformer to convert categorical features (treated as strings)
to weight-of-evidence values for the different category levels.

The formula used for computing w.o.e. is a slightly modified
version of what is found in this StackOverflow answer. To the
referenced formula, we have added an adjustment/smoothing constant
0.5 to the numerator and constant 1.0 to the denominator to avoid
log(0) and division by zero.

[Ref.] https://stackoverflow.com/questions/60892714/
how-to-get-the-weight-of-evidence-woe-and-information-value-iv-in-python/60892828#60892828
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Dict, List


class Woe(TransformerMixin, BaseEstimator):
    def __init__(self, col_names: np.ndarray = None, missing_val: str = '0'):
        """
        col_names: A list of column names for the data which would be fit and
                   transformed using this transformer.
        missing_val: A string which defines the placeholder for missing values.
        """
        self.col_names: np.ndarray = col_names

        self.missing_val: str = missing_val

        self.information_values: np.ndarray = np.zeros(len(col_names)) * np.nan

        self._woe_dict: Dict[str, pd.DataFrame] = dict()

        self._woe_mv: Dict[str, float] = dict()

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:

        x_df: pd.DataFrame = pd.DataFrame(X.astype(str), columns=self.col_names)

        y_df: pd.DataFrame = pd.DataFrame(y.astype(int), columns=['label'])

        # Hack to gracefully handle missing value in every feature.
        # Adding dummy rows to X and y. For large enough datasets,
        # this wouldn't make any different in classifier training.
        m, n = x_df.shape

        x_df.loc[m] = [self.missing_val] * n

        y_df.loc[m] = 0

        # Now fit the data
        for i, col in enumerate(self.col_names):
            woe_df: pd.DataFrame = pd.crosstab(x_df[col], y_df.label)

            neg_sum, pos_sum = woe_df.sum(0)

            # Computing weight of evidence for all the levels of the current category
            # by adding an adjustment/smoothing constant to the actual frequencies.
            woe_df = woe_df.assign(
                woe=lambda dfx: np.log((dfx[1] + 0.5) / (pos_sum + 1.)) - np.log((dfx[0] + 0.5) / (neg_sum + 1.)),
                iv=lambda dfx: np.sum(dfx['woe'] * ((dfx[1] + 0.5) / (pos_sum + 1.) - (dfx[0] + 0.5) / (neg_sum + 1.))))
            self._woe_dict[col] = woe_df

            # Remember the w.o.e. of missing value for this column.
            # This will be used to lookup w.o.e. of any unseen value
            # in the transform() method.
            self._woe_mv[col] = woe_df.woe.loc[self.missing_val]

            self.information_values[i] = woe_df.iv.iloc[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x_df: pd.DataFrame = pd.DataFrame(X.astype(str), columns=self.col_names)
        feature_cols: List[str] = list()

        for col in self.col_names:
            woe_col: str = '_'.join(['woe', col])

            # Weight of evidence of any new/hitherto unseen values will appear as np.nan
            #            transform_with_nan = self._woe_dict[col].woe[x_df[col].values].values.astype(float)
            woe_df = self._woe_dict[col].woe.reindex(index=x_df[col].values)

            transform_with_nan = woe_df.values.astype(float)

            transform_wo_nan = np.where(np.isnan(transform_with_nan),
                                        self._woe_mv[col],
                                        transform_with_nan)

            x_df = x_df.assign(**{woe_col: transform_wo_nan.astype(float)})

            feature_cols.append(woe_col)

        return x_df[feature_cols].values