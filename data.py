import pandas as pd

"""
External calls to load data.
"""
def load_pbp_data(year: int):
    """
    Load NFL play by play data
    """
    BASE_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz?raw=True'
    
    if type(year) is not int:
        raise TypeError('Please provide an integer between 1999 and 2024 for the year argument.')

    if year < 1999 or year > 2024:
        raise SeasonNotFoundError('Play by play data is only available from 1999 to 2024.')

    df = pd.read_csv(BASE_URL.format(year=year), compression='gzip', low_memory=False)
    df.to_csv(f'./data/play_by_play_{year}.csv', index=False)
    return df

def load_season_schedule(years: list[int]) -> pd.DataFrame:
    validate_year_range(years)
    
    df = pd.read_csv(r'http://www.habitatring.com/games.csv')   
    df.to_csv(f'./data/season_schedule.csv', index=False)
    return df[df['season'].isin(years)]

def validate_year_range(years: list[int]) -> None:
    if not isinstance(years, list):
        raise ValueError('Input must be list of years.')
    
    if min(years) < 1999 or max(years) > 2024:
        raise SeasonNotFoundError('Data is only available from 1999 to 2024.')

"""
Wrapper functions with local cache.
"""
def get_pbp_data(year: int):
    """
    Checks if local copy of play by play data exists, if not, downloads it
    """
    try:
        df = pd.read_csv(f'./data/play_by_play_{year}.csv', low_memory=False)
    except FileNotFoundError:
        df = load_pbp_data(year)
    return df

def get_season_schedule(years: list[int]):
    """
    Checks if local copy of season schedule data exists, if not, downloads it
    """
    validate_year_range(years)
    try:
        df = pd.read_csv(f'./data/season_schedule.csv', low_memory=False)
        df = df[df['season'].isin(years)]
    except FileNotFoundError:
        df = load_season_schedule(years)
    return df
    
"""
Custom Exceptions.
"""

class SeasonNotFoundError(Exception):
    pass