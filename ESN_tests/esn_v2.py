"""ESN variant for ESN_tests: adds configurable washout and optional
warm-start. Duplicated from ../src/esn.py, not imported."""
import numpy as np
from sklearn.linear_model import Ridge


class EchoStateNetworkV2:
    def __init__(self, n_inputs, n_reservoir=200, spectral_radius=0.9,
                 sparsity=0.9, leak_rate=0.3, ridge_alpha=1.0, seed=0,
                 warm_start=False, warm_start_steps=100):
        """warm_start: if True, primes the reservoir with the sequence's own
        first input repeated `warm_start_steps` times, instead of zero.
        Fully causal, no peeking."""
        rng = np.random.default_rng(seed)
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate
        self.warm_start = warm_start
        self.warm_start_steps = warm_start_steps

        self.W_in = rng.uniform(-1, 1, size=(n_reservoir, n_inputs))

        W = rng.uniform(-1, 1, size=(n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) < sparsity
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius)

        self.readout = Ridge(alpha=ridge_alpha)

    def _initial_state(self, U):
        if not self.warm_start:
            return np.zeros(self.n_reservoir)
        x = np.zeros(self.n_reservoir)
        u0 = U[0]
        for _ in range(self.warm_start_steps):
            pre = self.W_in @ u0 + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre)
        return x

    def _run_reservoir(self, U):
        T = U.shape[0]
        states = np.zeros((T, self.n_reservoir))
        x = self._initial_state(U)
        for t in range(T):
            pre_activation = self.W_in @ U[t] + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
            states[t] = x
        return states

    def fit(self, sequences, targets, washout=5):
        X_parts, y_parts = [], []
        for U, target in zip(sequences, targets):
            states = self._run_reservoir(U)
            X_parts.append(states[washout:])
            y_parts.append(target[washout:])
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self.readout.fit(X, y)

    def predict_last(self, sequences):
        preds = []
        for U in sequences:
            states = self._run_reservoir(U)
            preds.append(self.readout.predict(states[-1:])[0])
        return np.array(preds)
