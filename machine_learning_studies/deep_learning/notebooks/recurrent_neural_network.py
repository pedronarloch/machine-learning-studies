# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

from machine_learning_studies.timeseries.data import syntethic

# %%
n_steps = 5
series = syntethic.generate_univariate_time_series(
    batch_size=1,
    n_steps=n_steps + 1,
)
