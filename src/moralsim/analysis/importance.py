import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

def compute_imp(
    X, y, groups, mode, scoring,
    n_splits=5, n_repeats=10, random_state=0
):
    """
    Returns:
      - importances: np.ndarray, shape (n_features,)
      - r2_train:  float, mean R² on the training folds
      - r2_oof:    float, mean R² on the validation folds
    """
    clf = RandomForestRegressor(n_estimators=100, random_state=random_state)

    gkf = GroupKFold(n_splits=n_splits)

    fold_imps    = []
    train_r2s    = []
    oof_r2s      = []
    train_mses   = []
    oof_mses     = []
    train_maes   = []
    oof_maes     = []
    
    # loop over folds
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)

        # evaluate R2
        y_pred_train = clf.predict(X_train)
        y_pred_test  = clf.predict(X_test)
        train_r2s.append(r2_score(y_train, y_pred_train))
        oof_r2s.append(r2_score(y_test,  y_pred_test))
        train_mses.append(mean_squared_error(y_train, y_pred_train))
        oof_mses.append(mean_squared_error(y_test, y_pred_test))
        train_maes.append(mean_absolute_error(y_train, y_pred_train))
        oof_maes.append(mean_absolute_error(y_test, y_pred_test))

        # permutation‐importance on the test fold
        res = permutation_importance(
            clf, X_test, y_test,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1
        )
        fold_imps.append(res.importances_mean)

    # aggregate results
    imp_arr = np.mean(fold_imps, axis=0)
    importances = pd.Series(imp_arr, index=X.columns)

    metrics = {
        "r2_train": np.mean(train_r2s),
        "r2_oof":   np.mean(oof_r2s),
        "mse_train":np.mean(train_mses),
        "mse_oof":  np.mean(oof_mses),
        "mse_mean": ((y - y.mean()**2)).mean(),
        "mae_train":np.mean(train_maes),
        "mae_oof":  np.mean(oof_maes),
    }

    for name, value in metrics.items():
        importances[name] = value

    return importances


def bootstrap_ci(
    X, df, groups, outcome_name, mode, scoring, factors,
    ci=0.95,
    n_boot=100, n_splits=5, n_repeats=5, random_state=0
):
    """
    Returns:
      - mean_series: pd.Series of means (importances + metrics)
      - ci_lower: dict mapping ci_level -> pd.Series of lower bounds
      - ci_upper: dict mapping ci_level -> pd.Series of upper bounds
    """
    # get template Series to infer index & length
    sample = compute_imp(
        X, df[outcome_name], groups, mode, scoring,
        n_splits=n_splits, n_repeats=n_repeats,
        random_state=random_state
    )
    index = sample.index
    n_metrics = len(index)
    boots = np.zeros((n_boot, n_metrics))
    run_ids = groups.unique()
    rng = np.random.RandomState(random_state)

    # bootstrap loop
    for i in range(n_boot):
        chosen = rng.choice(run_ids, size=len(run_ids), replace=True)
        boot_df = pd.concat([df[df["run"] == r] for r in chosen])
        Xb = pd.get_dummies(boot_df[factors], drop_first=True).reindex(columns=X.columns, fill_value=0)
        yb = boot_df[outcome_name]
        gb = boot_df["run"]
        boots[i, :] = compute_imp(
            Xb, yb, gb, mode, scoring,
            n_splits=n_splits, n_repeats=n_repeats,
            random_state=rng.randint(0, 1_000_000)
        ).values

    # aggregate
    mean_vals = boots.mean(axis=0)
    mean_series = pd.Series(mean_vals, index=index)

    ci_lower = {}
    ci_upper = {}
    alpha = (1 - ci) / 2
    lo_pct, hi_pct = 100 * alpha, 100 * (1 - alpha)
    lo_vals, hi_vals = np.percentile(boots, [lo_pct, hi_pct], axis=0)
    ci_lower = pd.Series(lo_vals, index=index)
    ci_upper = pd.Series(hi_vals, index=index)

    return mean_series, ci_lower, ci_upper