import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from .EDAs import mission_1_EDA, jisu_EDA_1
from .EDAs import age_0413_ver1, age_0413_ver2, age_0413_ver4, category_0414_ver1
from .EDAs import dohyun_0415_ver1, dohyun_0415_ver4, dohyun_0417_ver1
from .EDAs import rating_mean_feature
from .EDAs import final
from sklearn.model_selection import StratifiedKFold
import os

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2, is_dl : bool = False): # default EDA
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
    
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.users_data)
    books = pd.read_csv(args.books_data)
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    if args.eda == 'default':
        idx, context_train, context_test = process_context_data(users, books, train, test)
    elif args.eda == 'mission1':
        idx, context_train, context_test = mission_1_EDA(users, books, train, test)
    elif args.eda == 'jisu':
        idx, context_train, context_test = jisu_EDA_1(users, books, train, test)
    elif args.eda == 'age_0413_ver1':
        idx, context_train, context_test = age_0413_ver1(users, books, train, test)
    elif args.eda == 'age_0413_ver2':
        idx, context_train, context_test = age_0413_ver2(users, books, train, test)
    elif args.eda == 'age_0413_ver4':
        idx, context_train, context_test = age_0413_ver4(users, books, train, test)
    elif args.eda == 'category_0414_ver1':
        idx, context_train, context_test = category_0414_ver1(users, books, train, test)
    elif args.eda == 'dohyun_0415_ver1':
        idx, context_train, context_test = dohyun_0415_ver1(users, books, train, test)
    elif args.eda == 'dohyun_0415_ver4':
        idx, context_train, context_test = dohyun_0415_ver4(users, books, train, test)
    elif args.eda == 'dohyun_0417_ver1':
        idx, context_train, context_test = dohyun_0417_ver1(users, books, train, test)
    elif args.eda == 'final':
        idx, context_train, context_test = final(users, books, train, test)

    users, context_train, context_test = rating_mean_feature(users, context_train, context_test)

    if args.select_feature != 9999:
        select_fe = pd.read_csv('/opt/ml/level1_bookratingprediction-recsys-13/feature_selection_result.csv')
        features = []

        for x in select_fe.loc[args.select_feature, 'features'].split("'"):
            if x not in ['[', ', ', ']']:
                features.append(x)

        drop_columns = ['category_mean', 'category_median', 'category_std', 'category_high_mean', 'category_high_median', 'category_high_std']
        features = list(set(features) - set(drop_columns))

        context_train = context_train[features + ['rating']]
        context_test = context_test[features + ['rating']]

    context_df = pd.concat([context_train, context_test])
    context_df = context_df.drop(columns='rating')

    field_dims = []
    for col in context_df.columns:
        field_dims.append(context_df[col].nunique())

    field_dims = np.array(field_dims)

    # if args.eda == 'jisu':
    #     field_dims = np.array([len(user2idx), len(isbn2idx),
    #                             6, len(idx['loc_city2idx']), len(idx['loc_country2idx']),
    #                             len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    
    # elif args.eda == 'dohyun_0415_ver1':
    #     field_dims = np.array([len(user2idx), len(isbn2idx),
    #                             6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                             len(idx['categoryhigh2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    # elif args.eda in ['dohyun_0415_ver4', 'final']:  # TODO : 추후 final에 대해서 field_dim 처리해줘야함!
    #     field_dims = np.array([len(user2idx), len(isbn2idx),
    #                             6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                             len(idx['category2idx']), len(idx['categoryhigh2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    # elif args.eda == 'dohyun_0417_ver1':
    #     field_dims = np.array([len(user2idx), len(isbn2idx),
    #                             6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                             len(idx['category2idx']), len(idx['categoryhigh2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)


    # else:
    #     field_dims = np.array([len(user2idx), len(isbn2idx),
    #                             6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                             len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    # train 데이터와 validation 데이터의 인덱스 분할
    train_idx, valid_idx = train_test_split(data['train'].index, test_size=args.test_size, random_state=args.seed, shuffle=True)

    # train 데이터와 validation 데이터의 인덱스를 각각 CSV 파일로 저장
    os.makedirs('./data_index', exist_ok=True)
    data['train'].loc[train_idx].to_csv(f'./data_index/context_data_train_index.csv', index=False)
    data['train'].loc[valid_idx].to_csv(f'./data_index/context_data_valid_index.csv', index=False)
    return data

def stratified_kfold(args, data):
    skf = StratifiedKFold(n_splits= args.k_fold, shuffle=True, random_state=args.seed)
    counts = 0
    data['X_train'] = []
    data['y_train'] = []
    data['X_valid'] = []
    data['y_valid'] = []
    for train_index, valid_index in skf.split(data['train'].drop(['rating'], axis=1),data['train']['rating']):
        data['X_train'].append(data['train'].drop(['rating'], axis=1).loc[train_index])
        data['y_train'].append(data['train']['rating'].loc[train_index])
        data['X_valid'].append(data['train'].drop(['rating'], axis=1).loc[valid_index])
        data['y_valid'].append(data['train']['rating'].loc[valid_index])
        
    return data

def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
