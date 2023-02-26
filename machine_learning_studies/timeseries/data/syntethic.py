import numpy as np


def generate_univariate_time_series(batch_size: int, n_steps: int):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + Wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise

    return series[..., np.newaxis].astype(np.float32)
