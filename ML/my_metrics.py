from autogluon.core.metrics import make_scorer
import numpy as np

# 1. Custom score function
def custom_score_func(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # A: Normalized RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    norm_rmse = rmse / (np.max(y_true) - np.min(y_true))
    A = norm_rmse

    # B: Pearson correlation coefficient
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        B = 0
    else:
        B = np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

    # 최종 Score 계산
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score

# 2. make_scorer로 래핑
custom_metric = make_scorer(
    name='custom_score',
    score_func=custom_score_func,
    greater_is_better=True  # Score가 클수록 좋음
)
