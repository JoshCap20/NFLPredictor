# ML for NFL Game Winner Predictions

Simple project adapted from various sources to predict the winner of NFL games using machine learning. This is a combination of Python scripts and Juypter notebooks.

## Using the Model

Make sure to install the requirements by running `pip install -r requirements.txt` in the Python environment you are using.

To first generate the model, run the `train.ipynb` notebook. This will generate a `clf.pkl` file that will be used later to predict the winner of games. Running this notebook may take a few minutes as it acquires the necessary data and trains the models.

Then, you can run the `predict.ipynb` notebook to predict the winner of games for a given week. This will then analyze betting lines and determine if there is a good bet to make.

`data.py` provides utilities for externally acquiring the data needed for the model, and `weekly_predictions.py` provides utilities for predicting the winner of games for a given week.
