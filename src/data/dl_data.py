import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from . import process_context_data
from .EDAs import mission_1_EDA, jisu_EDA_1
from .EDAs import age_0413_ver1, age_0413_ver2, age_0413_ver4, category_0414_ver1
from .EDAs import dohyun_0415_ver1, dohyun_0415_ver4

def dl_data_load(args):
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

    # EDA 방식 결정
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

    if args.eda == 'jisu':
        field_dims = np.array([len(user2idx), len(isbn2idx),
                                6, len(idx['loc_city2idx']), len(idx['loc_country2idx']),
                                len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    
    elif args.eda == 'dohyun_0415_ver1':
        field_dims = np.array([len(user2idx), len(isbn2idx),
                                6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                                len(idx['categoryhigh2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    elif args.eda == 'dohyun_0415_ver4':
        field_dims = np.array([len(user2idx), len(isbn2idx),
                                6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                                len(idx['category2idx']), len(idx['categoryhigh2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    else:
        field_dims = np.array([len(user2idx), len(isbn2idx),
                                6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                                len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    # field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

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

def dl_data_split(args, data):
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
    return data

def dl_data_loader(args, data):
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
