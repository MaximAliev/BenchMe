from collections import Counter
import itertools
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split as tts
from typing import cast



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

def infer_positive_class_label(y_train: Union[pd.Series, pd.DataFrame]) -> str:
    class_belongings = Counter(y_train)
    if len(class_belongings) > 2:
        raise ValueError("Multiclass problems currently not supported =(.")

    class_belongings_formatted = '; '.join(f"{k}: {v}" for k, v in class_belongings.items())
    logger.debug(f"Class belongings: {{{class_belongings_formatted}}}")

    class_belongings_iterator = iter(sorted(cast(Iterable, class_belongings)))
    *_, pos_label = class_belongings_iterator
    logger.debug(f"Inferred positive class label: {pos_label}.")

    number_of_positives = class_belongings.get(pos_label)
    if number_of_positives is None:
        raise ValueError("Unknown positive class label.")
    
    return pos_label