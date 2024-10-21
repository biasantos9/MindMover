
# Imports
import numpy as np
from typing import Tuple


def filter_by_condition(X: np.ndarray, Y: np.ndarray, condition: str) -> tuple:
   
    if not condition:
        raise ValueError("You have to select the conditions!")

    condition_upper = condition.upper()

    if condition_upper == "ALL":
        return X, Y
    else:
        if condition_upper in {"PRON", "PRONOUNCED"}:
            p = 0
        elif condition_upper in {"IN", "INNER"}:
            p = 1
        elif condition_upper in {"VIS", "VISUALIZED"}:
            p = 2
        else:
            raise ValueError(f"The condition '{condition}' doesn't exist!")

        X_r = X[Y[:, 2] == p]
        Y_r = Y[Y[:, 2] == p]

    return X_r, Y_r



def filter_by_class(X: np.ndarray, Y: np.ndarray, class_condition: str) -> tuple:
    
    if not class_condition:
        raise ValueError("You have to select the classes for each condition!")

    class_condition_upper = class_condition.upper()

    if class_condition_upper == "ALL":
        return X, Y
    else:
        if class_condition_upper in {"UP", "ARRIBA"}:
            p = 0
        elif class_condition_upper in {"DOWN", "ABAJO"}:
            p = 1
        elif class_condition_upper in {"RIGHT", "DERECHA"}:
            p = 2
        elif class_condition_upper in {"LEFT", "IZQUIERDA"}:
            p = 3
        else:
            raise ValueError(f"The class '{class_condition}' doesn't exist!")

        X_r = X[Y[:, 1] == p]
        Y_r = Y[Y[:, 1] == p]

    return X_r, Y_r

