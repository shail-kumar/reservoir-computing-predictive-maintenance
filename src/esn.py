"""Classical Echo State Network (ESN).

Core ESN recipe: a fixed random reservoir with the echo-state property,
driven by the raw sensor sequence, with only a ridge-regression readout
trained on the reservoir states. Simpler than the closest literature
reference, Rigamonti et al. (PHM Society, 2016) - no output feedback, no
elbow-point segmentation, no hand-selected/synthetic signals - see
WRITEUP.md for what that reference actually is and isn't used for here.
"""

import numpy as np
from sklearn.linear_model import Ridge


class EchoStateNetwork:
    def __init__(
        self,
        n_inputs,
        n_reservoir=200,
        spectral_radius=0.9,
        sparsity=0.9,
        leak_rate=0.3,
        ridge_alpha=1.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate

        self.W_in = rng.uniform(-1, 1, size=(n_reservoir, n_inputs))

        W = rng.uniform(-1, 1, size=(n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) < sparsity
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius)

        self.readout = Ridge(alpha=ridge_alpha)

    def _run_reservoir(self, U):
        """U: (T, n_inputs) -> states: (T, n_reservoir)."""
        T = U.shape[0]
        states = np.zeros((T, self.n_reservoir))
        x = np.zeros(self.n_reservoir)
        for t in range(T):
            pre_activation = self.W_in @ U[t] + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
            states[t] = x
        return states

    def fit(self, sequences, targets, washout=5):
        """sequences: list of (T_i, n_inputs) arrays.
        targets: list of (T_i,) RUL arrays, aligned with sequences.
        washout: initial timesteps dropped per sequence so the reservoir's
        state reflects its own driven dynamics rather than the zero
        initial condition.
        """
        X_parts, y_parts = [], []
        for U, target in zip(sequences, targets):
            states = self._run_reservoir(U)
            X_parts.append(states[washout:])
            y_parts.append(target[washout:])
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self.readout.fit(X, y)

    def predict_last(self, sequences):
        """Predict RUL at the final observed cycle of each sequence - the
        only data available at prediction time for a truncated test
        trajectory."""
        preds = []
        for U in sequences:
            states = self._run_reservoir(U)
            preds.append(self.readout.predict(states[-1:])[0])
        return np.array(preds)
