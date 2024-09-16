"""
Utility functions for running weekly predictions, used in `predict.ipynb`.
"""

from data import get_season_schedule, get_pbp_data
import pandas as pd
import numpy as np
import joblib
import itertools


def load_weekly_games(year: int, week: int) -> pd.DataFrame:
    """
    Load weekly schedule for a given week and year.
    """
    return get_season_schedule([year]).query(f"week == {week}")


def load_pbp_data(year: int) -> pd.DataFrame:
    """
    Load play by play data for a given year.
    """
    return get_pbp_data(year)


def load_model(model_name: str) -> any:
    """
    Load model from disk. Assumes model and model directory exist.
    """
    if model_name not in ["clf", "ensemble", "stacking"]:
        raise ValueError(
            "Invalid model name. Must be one of ['clf', 'ensemble', 'stacking']"
        )
    return joblib.load(f"./models/{model_name}.joblib")


def evaluate_game(home_team: str, away_team: str, model: any, data: pd.DataFrame):
    def ewma(data, window):
        """
        Calculate the most recent value for EWMA given an array of data and a window size
        """
        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha
        scale = 1 / alpha_rev
        n = data.shape[0]
        r = np.arange(n)
        scale_arr = scale**r
        offset = data[0] * alpha_rev ** (r + 1)
        pw0 = alpha * alpha_rev ** (n - 1)
        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out[-1]

    offense = data.loc[(data["posteam"] == home_team) | (data["posteam"] == away_team)]
    defense = data.loc[(data["defteam"] == home_team) | (data["defteam"] == away_team)]

    rushing_offense = (
        offense.loc[offense["rush_attempt"] == 1]
        .groupby(["posteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team"})
    )
    passing_offense = (
        offense.loc[offense["pass_attempt"] == 1]
        .groupby(["posteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team"})
    )
    rushing_defense = (
        defense.loc[defense["rush_attempt"] == 1]
        .groupby(["defteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team"})
    )
    passing_defense = (
        defense.loc[defense["pass_attempt"] == 1]
        .groupby(["defteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team"})
    )

    super_bowl_X = np.zeros(8)

    for i, (tm, stat_df) in enumerate(
        itertools.product(
            [home_team, away_team],
            [rushing_offense, passing_offense, rushing_defense, passing_defense],
        )
    ):
        ewma_value = ewma(stat_df.loc[stat_df["team"] == tm]["epa"].values, 20)
        super_bowl_X[i] = ewma_value

    predicted_winner = model.predict(super_bowl_X.reshape(1, 8))[0]
    predicted_proba = model.predict_proba(super_bowl_X.reshape(1, 8))[0]

    winner = home_team if predicted_winner else away_team
    win_prob = predicted_proba[-1] if predicted_winner else predicted_proba[0]

    return winner, win_prob


def run_weekly_predictions(
    year: int, week: int, model_name: str = "clf"
) -> pd.DataFrame:
    """
    Run weekly predictions for a given year and week.
    """
    # Load data
    games: pd.DataFrame = load_weekly_games(year, week)
    pbp_data: pd.DataFrame = load_pbp_data(year)

    # Load model
    model = load_model(model_name)

    # Run predictions
    output_df = games.copy()
    output_df["predicted_winner"], output_df["win_probability"] = zip(
        *games.apply(
            lambda row: evaluate_game(
                row["home_team"], row["away_team"], model, pbp_data
            ),
            axis=1,
        )
    )

    # Output predictions
    output_df.to_csv(f"./data/weekly_predictions_{year}_{week}.csv", index=False)
    return output_df


if __name__ == "__main__":
    results = run_weekly_predictions(2024, 3)
    print(results)
