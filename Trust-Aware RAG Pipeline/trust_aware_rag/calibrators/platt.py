"""
Platt Scaling Calibrator & Calibration Metrics.

Fits a base classifier on the training set, fits Platt Scaling on validation logits,
and provides standard Expected Calibration Error (ECE) metric computation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Fits base linear classifier on Train Set, then fits Platt Scaling on Validation Set.
    Computes Expected Calibration Error (ECE) and calibrated probabilities.
    """

    def __init__(self):
        self.base_classifier = LogisticRegression(C=1.0, random_state=42)
        self.platt_scaler = LogisticRegression(C=1.0, random_state=42)
        self.is_fitted = False

    def fit_base(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.base_classifier.fit(X_train, y_train)

    def fit_platt_scaling(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        val_logits = self.base_classifier.decision_function(X_val).reshape(-1, 1)
        self.platt_scaler.fit(val_logits, y_val)
        self.is_fitted = True

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.base_classifier.decision_function(X).reshape(-1, 1)
        return self.platt_scaler.predict_proba(logits)[:, 1]

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE) with equal-width probability bins."""
        if len(y_true) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

            bin_count = np.sum(mask)
            if bin_count > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

        return float(ece)


class CascadedCalibrator:
    """
    Learns and calibrates a combined probability score on validation data
    combining Stage 1 continuous logits and Stage 2 discrete judge flags.
    Mathematically valid multi-signal calibration adhering to proper validation separation.
    """

    def __init__(self):
        self.scaler = LogisticRegression(C=1.0, random_state=42)
        self.is_fitted = False

    def fit(self, s1_logits: np.ndarray, s2_flags: np.ndarray, y_val: np.ndarray) -> None:
        s1 = np.asarray(s1_logits).reshape(-1, 1)
        s2 = np.asarray(s2_flags).reshape(-1, 1)
        X = np.column_stack([s1, s2])
        self.scaler.fit(X, y_val)
        self.is_fitted = True

    def predict_proba(self, s1_logits: np.ndarray, s2_flags: np.ndarray) -> np.ndarray:
        s1 = np.asarray(s1_logits).reshape(-1, 1)
        if not self.is_fitted:
            return (1.0 / (1.0 + np.exp(-s1))).ravel()
        s2 = np.asarray(s2_flags).reshape(-1, 1)
        X = np.column_stack([s1, s2])
        return self.scaler.predict_proba(X)[:, 1]

