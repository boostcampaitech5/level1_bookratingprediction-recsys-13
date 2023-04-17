import numpy as np
import pandas as pd

def rating_mean_feature(users : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
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
    # 사용자별 평점 평균 분산 미디안

    train_df_means = ratings1.groupby('user_id')['rating'].mean()
    train_df_total_mean = ratings1['rating'].mean()
    train_df_vars = ratings1.groupby('user_id')['rating'].var()
    train_df_total_var = ratings1['rating'].var()
    train_df_medians = ratings1.groupby('user_id')['rating'].median()
    train_df_total_median = ratings1['rating'].median()

    users['user_rating_avg'] = users['user_id'].apply(lambda x: train_df_means[x] if x in train_df_means else train_df_total_mean)
    users['user_rating_var'] = users['user_id'].apply(lambda x: train_df_vars[x] if x in train_df_vars else train_df_total_var).fillna(0)
    users['user_rating_median'] = users['user_id'].apply(lambda x: train_df_medians[x] if x in train_df_medians else train_df_total_median)

    ratings1['user_rating_avg'] = ratings1['user_id'].apply(lambda x: train_df_means[x] if x in train_df_means else train_df_total_mean)
    ratings1['user_rating_var'] = ratings1['user_id'].apply(lambda x: train_df_vars[x] if x in train_df_vars else train_df_total_var).fillna(0)
    ratings1['user_rating_median'] = ratings1['user_id'].apply(lambda x: train_df_medians[x] if x in train_df_medians else train_df_total_median)

    ratings2['user_rating_avg'] = ratings2['user_id'].apply(lambda x: train_df_means[x] if x in train_df_means else train_df_total_mean)
    ratings2['user_rating_var'] = ratings2['user_id'].apply(lambda x: train_df_vars[x] if x in train_df_vars else train_df_total_var).fillna(0)
    ratings2['user_rating_median'] = ratings2['user_id'].apply(lambda x: train_df_medians[x] if x in train_df_medians else train_df_total_median)
    
    if is_dl:
        return users, books
    return users, ratings1, ratings2