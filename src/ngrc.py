"""Next-generation reservoir computing (NG-RC) - the primary/novel model.

No random reservoir and no washout phase: 'memory' comes entirely from an
explicit delay embedding of the raw sensor history, which is then expanded
with polynomial features and mapped to RUL with a linear (ridge) readout
(Gauthier et al., 2021, "Next generation reservoir computing").

Uses sklearn's PolynomialFeatures for the nonlinear expansion rather than
hand-rolling it - that part is well-trodden, well-tested code; the actual
technique being demonstrated is the delay-embedding + linear-readout
formulation itself, not polynomial expansion mechanics.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class NGRC:
    def __init__(
        self, n_lags=4, degree=2, ridge_alpha=1.0, solver="lsqr", dtype=np.float32
    ):
        """solver='lsqr': iterative least-squares, uses only matrix-vector
        products against the design matrix - never forms the
        (features x features) normal-equations matrix the default Cholesky
        solver does. Drops memory from rows*features + features**2 to just
        rows*features, enabling larger n_lags/degree combinations.

        tol/max_iter tightened from sklearn's defaults (1e-4, 1000): the
        default under-converges (RMSE off by ~0.09 vs. Cholesky on a known
        config); tol=1e-6 matches Cholesky to within 0.00007 RMSE.

        dtype=float32 by default for memory; pass float64 when precision
        matters more than memory (e.g. configs small enough that the
        memory saving is moot).
        """
        self.n_lags = n_lags
        self.dtype = dtype
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        # copy_X=False: Ridge's default (True) allocates a second full copy of X to
        # mean-center it (fit_intercept defaults to True) - doubling memory on top of
        # fit()'s own centering below is what causes an OOM at scale. Safe to center in
        # place; the pre-fit X is never needed again.
        if solver == "lsqr":
            self.readout = Ridge(
                alpha=ridge_alpha, solver=solver, tol=1e-6, max_iter=10000, copy_X=False
            )
        else:
            self.readout = Ridge(alpha=ridge_alpha, solver=solver, copy_X=False)
        self._poly_fitted = False

    def _delay_embed(self, U):
        """U: (T, n_inputs) -> (T - n_lags + 1, n_inputs * n_lags); each row
        concatenates the current reading with the n_lags-1 preceding ones.
        """
        T = U.shape[0]
        rows = [
            U[t - self.n_lags + 1 : t + 1].flatten() for t in range(self.n_lags - 1, T)
        ]
        return np.array(rows, dtype=self.dtype)

    def fit(self, sequences, targets):
        """sequences: list of (T_i, n_inputs) arrays.
        targets: list of (T_i,) RUL arrays, aligned with sequences.

        Stacks the narrow pre-expansion arrays once, then calls
        poly.fit_transform on the full stack - avoids transforming
        per-sequence into a list and vstacking that, which briefly holds
        two full-size (rows x features) copies at once and roughly doubles
        peak memory at large feature counts.

        dtype (float32 by default, see __init__): the design matrix can be
        too large in absolute terms at float64 even without duplication;
        float32 halves memory directly.
        """
        embedded_all = [self._delay_embed(U) for U in sequences]
        embedded_stacked = np.vstack(embedded_all)

        X = self.poly.fit_transform(embedded_stacked).astype(self.dtype, copy=False)
        self._poly_fitted = True
        y = np.concatenate([target[self.n_lags - 1 :] for target in targets]).astype(
            self.dtype
        )

        self.readout.fit(X, y)

    def predict_last(self, sequences):
        """Predict RUL at the final observed cycle of each sequence."""
        preds = []
        for U in sequences:
            embedded = self._delay_embed(U)
            X = self.poly.transform(embedded[-1:])
            preds.append(self.readout.predict(X)[0])
        return np.array(preds)
