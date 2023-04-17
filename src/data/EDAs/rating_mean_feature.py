import numpy as np
import pandas as pd

def rating_mean_feature(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame) -> tuple:
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """
    return users, books, ratings1, ratings1