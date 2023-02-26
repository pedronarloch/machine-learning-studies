# machine-learning-studies

This repository contains resources related to machine learning algorithms
and toy-problem examples.

## How to set up
- run `poetry install` to install all dependencies.
- run `pre-commit install` to install the pre-commit hooks.

## How to create Jupyter Notebooks from Sources

In this repo we use `jupytext` to create our running examples.
One of the biggest benefits of using `.py` instead of `.ipynb` is
the benefits of IDE, linting, and pre-commit hooks.

If you want to, you can render the files into `.ipynb` for better visualization as follwing:
- Activate your `.venv` typing `poetry shell` in your terminal.
- With the activated environment, type `jupytext --to notebook <desired_script.py>`
