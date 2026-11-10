
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    total_actual = np.sum(np.abs(y_true))

    wape = (
        np.sum(np.abs(y_true - y_pred)) / total_actual * 100
        if total_actual > 0 else np.nan
    )

    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) > 1 else np.nan
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "R2": r2,
    }
