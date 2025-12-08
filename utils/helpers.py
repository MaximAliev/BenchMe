import itertools
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split as tts


def make_imbalanced(
    self,
    X_train,
    y_train,
    class_belongings,
    pos_label
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        is_dataset_initially_imbalanced = True
        number_of_positives = class_belongings.get(pos_label)
        proportion_of_positives = number_of_positives / len(y_train)

        # For extreme case - 0.01, for moderate - 0.2, for mild - 0.4.
        if proportion_of_positives > 0.01:
            coefficient = 0.01
            updated_number_of_positives = int(coefficient * len(y_train))

            assert updated_number_of_positives < 10, "Number of positive class instances is too low."
            class_belongings[pos_label] = updated_number_of_positives
            is_dataset_initially_imbalanced = False

        if not is_dataset_initially_imbalanced:
            X_train, y_train = make_imbalance(
                X_train,
                y_train,
                sampling_strategy=class_belongings)
            logger.debug("Imbalancing applied.")

        return X_train, y_train

def split_data_on_train_and_test(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> List[Union[pd.DataFrame, pd.Series]]:
        if y is not None:
            return tts(
                X,
                y,
                random_state=42,
                test_size=0.2,
                stratify=y)
        else:
             return tts(
                X,
                y,
                random_state=42,
                test_size=0.2)
