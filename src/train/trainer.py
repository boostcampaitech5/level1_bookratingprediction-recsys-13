import os
import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam
import pickle

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor

from torch.utils.tensorboard import SummaryWriter ###

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        pass
    
    writer = SummaryWriter(log_dir=os.getcwd() + f'/log/{setting.save_time}_{args.model}', filename_suffix='tensorboard_logs') ###

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1
        valid_loss = valid(args, model, dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        writer.add_scalar("Loss/train", total_loss/batch, epoch+1) ###
        writer.add_scalar("Loss/valid", valid_loss, epoch+1) ###
        
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    logger.close()
    writer.flush() ###
    writer.close() ###
    return model

def stratified_kfold(args, data, n):
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=args.seed)
    counts = 0
    for train_index, valid_index in skf.split(data['train'].drop(['rating'], axis=1),data['train']['rating']):
        if counts == n:
            data['X_train'], data['y_train'] = data['train'].drop(['rating'], axis=1).loc[train_index], data['train']['rating'].loc[train_index]
            data['X_valid'], data['y_valid'] = data['train'].drop(['rating'], axis=1).loc[valid_index], data['train']['rating'].loc[valid_index]
            break
        else:
            counts += 1
        
    return data

def gbdt_train(args, model, data, logger, setting):

    evals = [(data['X_valid'],data['y_valid'])]
    if args.model == 'catboost':
        if args.eda == 'jisu':
            cat_features = ['user_id', 'isbn', 'category', 'publisher', 'language', 'book_author','age','location_city','location_country']
        elif args.eda == 'category_0414_ver1':
            cat_features = ['user_id', 'isbn', 'category_high', 'category', 'publisher', 'language', 'book_author','age','location_city', 'location_state', 'location_country']
        elif args.eda == '0415_dohyun_ver1':
            cat_features = ['user_id', 'isbn', 'category_high', 'publisher', 'language', 'book_author','age','location_city', 'location_state', 'location_country']
        else:
            cat_features = ['user_id', 'isbn', 'category', 'publisher', 'language', 'book_author','age','location_city','location_state','location_country']
            
        cat_features = list(set(cat_features).intersection(list(data['X_train'].columns)))
        for i in  range(args.k_fold):
            data = stratified_kfold(args, data, i)
            model.fit(data['X_train'], data['y_train'], eval_set= evals, early_stopping_rounds=300, cat_features=cat_features, verbose=100)
            save_model_pkl(args, model, setting, i)

        model.fit(data['X_train'], data['y_train'], eval_set= evals, early_stopping_rounds=300, cat_features=cat_features, verbose=100)
    elif args.model == 'lgbm':
        for i in  range(args.k_fold):
            data = stratified_kfold(args, data, i)
            model.fit(data['X_train'], data['y_train'], eval_metric=args.loss_fn, eval_set=evals, verbose=100)
            save_model_pkl(args, model, setting, i)
    elif args.model == 'xgb':
        for i in  range(args.k_fold):
            data = stratified_kfold(args, data, i)
            model.fit(data['X_train'], data['y_train'], eval_metric=args.loss_fn, eval_set=evals, verbose=100)
            save_model_pkl(args, model, setting, i)
    os.makedirs(args.saved_model_path, exist_ok=True)
    
    # torch.save(torch.jit.script(model), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    # logger.log(model.get_all_params())
    logger.close()
    return model

def save_model_pkl(args, model, setting, i):
    with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model{i+1}.pkl', 'wb') as f:
        pickle.dump(model, f)

def select_feature(args, model, data):
    
    X_train, X_valid, y_train, y_valid = data['X_train'].copy(), data['X_valid'].copy(), data['y_train'].copy(), data['y_valid'].copy()
    
    features = list(X_train.columns)
    feature_list = []
    experiment_result = pd.DataFrame({'features':['0']*len(features), 'len_features':['0']*len(features), 'rmse':np.zeros(len(features))})
    features_copy = features.copy()
    model_copy = CatBoostRegressor(iterations=100, learning_rate=args.lr, random_state=args.seed, eval_metric=args.loss_fn, task_type="GPU")

    for i in tqdm.tqdm(range(len(features)), desc='selecting features...'):
        evals = [(X_valid[features_copy], y_valid)]
        if args.model == 'catboost':
            cat_features = ['user_id', 'isbn', 'category_high', 'category', 'publisher', 'language', 'book_author','age','location_city','location_state','location_country']
            cat_features = list(set(cat_features).intersection(list(X_train[features_copy].columns)))
            model_copy.fit(X_train[features_copy], y_train, eval_set=evals, early_stopping_rounds=300, cat_features=cat_features, verbose=0)
        elif args.model == 'lgbm':
            model_copy.fit(X_train[features_copy], y_train, eval_metric=args.loss_fn, eval_set=evals, verbose=0)
    
        result = permutation_importance(model_copy, X_train[features_copy], y_train, 
                                        scoring = make_scorer(mean_squared_error ,greater_is_better=False),
                                        n_repeats=10,
                                        random_state=args.seed)
        
        sorted_result = result.importances_mean.argsort()
        importances = pd.DataFrame(result.importances_mean[sorted_result], index=X_train[features_copy].columns[sorted_result]).sort_values(0, ascending=False)   
        importances = importances.rename(columns={0:'importances'})
        
        feature = list(importances.index)[0]
        features_copy.remove(feature)
        
        feature_list.append(feature)
        
        temp_model = model
        temp_evals = [(X_valid[feature_list], y_valid)]
        
        if args.model == 'catboost':
            temp_cat_features = ['user_id', 'isbn', 'category_high', 'category', 'publisher', 'language', 'book_author','age','location_city','location_state','location_country']
            temp_cat_features = list(set(temp_cat_features).intersection(list(X_train[feature_list].columns)))
            temp_model.fit(X_train[feature_list], y_train, eval_set=temp_evals, early_stopping_rounds=300, cat_features=temp_cat_features, verbose=0)
        elif args.model == 'lgbm':
            temp_model.fit(X_train[feature_list], y_train, eval_metric=args.loss_fn, eval_set=temp_evals, verbose=0)
        
        y_pred = temp_model.predict(X_valid[feature_list])
        RMSE = mean_squared_error(y_valid, y_pred) ** (0.5)

        experiment_result.loc[i, 'len_features'] = len(feature_list)
        experiment_result.loc[i, 'features'] = str(feature_list)
        experiment_result.loc[i, 'rmse'] = RMSE
    
    experiment_result = experiment_result.sort_values('rmse').reset_index(drop=True)

    print(experiment_result)
    
    features_idx = input('SELECT Feature index : ')
    features_idx = float(features_idx)
    
    features = []
    for x in experiment_result.loc[int(features_idx), 'features'].split("'"):
        if x not in ['[', ', ', ']']:
            features.append(x)

    print(f'{features} selected!')

    data['X_train'] = data['X_train'][features]
    data['X_valid'] = data['X_valid'][features]
    data['test'] = data['test'][features]
    tr = pd.concat([data['X_train'], pd.DataFrame({'rating':data['y_train']})], axis=1)
    val = pd.concat([data['X_valid'], pd.DataFrame({'rating':data['y_valid']})], axis=1)
    data['train'] = pd.concat([tr, val])
    del tr, val
    
    return data

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

def gbdt_test(args, model, data, setting):
    predicts_list = list()
    for i in range(args.k_fold):
        if args.use_best_model == True:
            with open(f'./saved_models/{setting.save_time}_{args.model}_model{i+1}.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            pass
        predicts_list.append(model.predict(data['test']))

    predicts = np.mean(predicts_list, axis=0)
                             
    return predicts
