import time
import argparse
import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')

from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader, stratified_kfold
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test, gbdt_train, gbdt_test, select_feature

def main(args):
    Setting.seed_everything(args.seed)


    ######################## DATA LOAD
    if not args.use_saved_data:
        print(f'--------------- {args.model} Load Data ---------------')
        if args.model in ('FM', 'FFM', 'catboost', 'lgbm', 'xgb', 'tabnet'):
            data = context_data_load(args)
        elif args.model in ('NCF', 'WDN', 'DCN'):
            data = dl_data_load(args)
        elif args.model == 'CNN_FM':
            data = image_data_load(args)
        elif args.model == 'DeepCoNN':
            import nltk
            nltk.download('punkt')
            data = text_data_load(args)
        else:
            pass
        
        # 강제적으로 저장한다.
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        with open(f'{args.data_path}/{save_time}_{args.model}_data.pt',"wb") as f:
            pickle.dump(data, f)
    else:
        # 저장 된 pickle 파일을 불러서 data에 넣는다.
        with open(args.saved_data,"rb") as f:
            data = pickle.load(f)

    if args.select_feature != (9999):
        select_fe = pd.read_csv('./feature_selection_result.csv')
        features = []

        for x in select_fe.loc[args.select_feature, 'features'].split("'"):
            if x not in ['[', ', ', ']']:
                features.append(x)

        drop_columns = ['category_mean', 'category_median', 'category_std', 'category_high_mean', 'category_high_median', 'category_high_std']
        features = list(set(features) - set(drop_columns))
        
        data['train'] = data['train'][features + ['rating']]
        data['test'] = data['test'][features]

    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)
        
    elif args.model in ('catboost', 'lgbm', 'xgb', 'tabnet'):
        if args.k_fold == 1:
            data = context_data_split(args, data)
        else:
            data = stratified_kfold(args, data)

    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    else:
        pass

    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args,data)

    ######################## Select Feature
    if (args.model in ('catboost', 'lgbm')) & ((args.FS)):
        print(f'--------------- SELECT FEATURES ---------------')
        select_feature(args, model, data)

    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    if args.model in ('catboost', 'lgbm', 'xgb', 'tabnet'):
        model = gbdt_train(args, model, data, logger, setting)
    else:
        model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    if args.model in ('catboost', 'lgbm', 'xgb', 'tabnet'):
        predicts = gbdt_test(args, model, data, setting)
    else:
        predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'xgb', 'lgbm', 'catboost', 'tabnet', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'):
        submission['rating'] = predicts
        submission.loc[submission['rating'] > 10, 'rating'] = 10
        submission.loc[submission['rating'] < 1, 'rating'] = 1
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'xgb', 'lgbm', 'catboost', 'tabnet', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE', 'rmse'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM', 'ADAMw'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

    ############### EDA Selection
    arg('--eda', type=str, help='user와 books에 대한 전처리 방식을 선택할 수 있습니다.')

    ############### K-FOLD
    arg('--k_fold', type=int, default=1, help='K-FOLD의 K값을 조정할 수 있습니다.')

    ############### Feature Selection
    arg('--FS', type=bool, default=False, help='변수 선택 단계를 거칠 것인지를 결정합니다.')
    arg('--select_feature', type=int, default=9999, help='변수 선택 결과에서 사용할 인덱스를 결정합니다. (0부터 시작)')

    ############### after eda file load
    arg('--users_data', type=str, default='/opt/ml/data/users.csv', help='Users data path를 설정할 수 있습니다.')
    arg('--books_data', type=str, default= '/opt/ml/data/books.csv', help='Books data path를 설정할 수 있습니다.')
    arg('--data_path', type=str, default='/opt/ml/data/', help='default Data path를 설정할 수 있습니다.')
    arg('--use_saved_data', type=bool, default=False, help='EDA가 끝난 데이터를 사용할지 설정합니다.')
    arg('--saved_data', type=str, default='/opt/ml/data/', help='EDA가 끝난 데이터 파일을 설정할 수 있습니다. (eg: /opt/ml/data/20230418_231549_catboost_data.pt)')

    args = parser.parse_args()
    main(args)
