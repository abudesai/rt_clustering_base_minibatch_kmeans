from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""

    def __init__(self, columns, selector_type="keep"):
        self.columns = columns
        self.selector_type = selector_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if self.selector_type == "keep":
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == "drop":
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        else:
            raise Exception(
                f"""
                Error: Invalid selector type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} """
            )
        return X
