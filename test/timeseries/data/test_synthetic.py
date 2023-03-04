import numpy as np

import machine_learning_studies.timeseries.data.syntethic as synthetic


def test_create_sliding_time_window():
    data = np.arange(start=1, stop=10)
    synthetic.create_sliding_time_window(
        data=data,
        window_size=2,
        output_size=1,
    )
