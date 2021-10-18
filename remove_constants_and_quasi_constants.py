from fast_ml.utilities import display_all
from fast_ml.feature_selection import get_constant_features
import pandas as pd
import numpy as np
import pandas_flavor as pf

@pf.register_dataframe_method
def constant_quasi_constant_feature_removal(df, threshold = 0.99):
    """
    Constant and quasi-constant function that searches the 
    feature space and removes all features that are either constant
    or quasi-constant based upon the user-specified threshold.
    Args:
        df ([Pandas Dataframe]): 
            A Pandas Dataframe that has the all of the features being 
            considered for actual models.
        threshold ([int]) optional): 
            Sets the variance threshold for features to select. 
            Defaults to 0.99.
    Returns:
        [Pandas Dataframe]: 
            A Pandas Dataframe that has removed all the constants
            and quasi-constants from the feature space. 
    """
    constant_features = get_constant_features(df, threshold=threshold, dropna=False)

    constant_features_list = constant_features['Var'].to_list()

    df.drop(columns=constant_features_list, inplace=True)

    return df
