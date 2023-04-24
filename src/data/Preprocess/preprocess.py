import re
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    
def replace_na(unique:np.array):
    unique = unique.tolist()
    unique.remove('na')
    unique = ['na'] + unique
    return unique

def year_of_publication_map(x):
    return round(x, -1)

def split_map(x:str) -> list:
    _str = str(x)
    _str_split = _str.split(',')
    if len(_str_split) >3:
        return [_str_split[0], '', _str_split[-1]]
    else:
        return _str_split

def rating_mean_feature(users : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
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
        return users
    return users, ratings1, ratings2

def age_0413_ver1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
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
    
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 5개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

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

    context_city = context_df[['location_city', 'age']].copy() ###
    context_city['location_city'] = context_city['location_city'].map(loc_city2idx) ###

    # train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    # train_df['age'] = train_df['age'].apply(age_map)
    # test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    # test_df['age'] = test_df['age'].apply(age_map)
    
    age_replace_idx = np.where(context_city.groupby('location_city')['age'].count() > 1)[0] ###
    age_replace = context_city.groupby('location_city')['age'].median()[age_replace_idx] ###
    age_replace_df = pd.DataFrame({'location_city':age_replace.index, 'age_fillna':age_replace.values}) ###

    train_df = train_df.merge(age_replace_df, on='location_city', how='left') ###
    train_df = train_df.drop(columns='age').rename(columns={'age_fillna':'age'}) ###
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean())) ###
    train_df['age'] = train_df['age'].apply(age_map) ###

    test_df = test_df.merge(age_replace_df, on='location_city', how='left') ###
    test_df = test_df.drop(columns='age').rename(columns={'age_fillna':'age'}) ###
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean())) ###
    test_df['age'] = test_df['age'].apply(age_map) ###

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

def age_0413_ver2(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
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

    context_city = context_df[['location_city', 'age']].copy() ###
    context_city['location_city'] = context_city['location_city'].map(loc_city2idx) ###

    # train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    # train_df['age'] = train_df['age'].apply(age_map)
    # test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    # test_df['age'] = test_df['age'].apply(age_map)
    
    age_replace_idx = np.where(context_city.groupby('location_city')['age'].count() > 1)[0] ###
    age_replace = context_city.groupby('location_city')['age'].median()[age_replace_idx] ###
    age_replace_df = pd.DataFrame({'location_city':age_replace.index, 'age_fillna':age_replace.values}) ###

    train_df = train_df.merge(age_replace_df, on='location_city', how='left') ###
    train_df = train_df.drop(columns='age').rename(columns={'age_fillna':'age'}) ###
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean())) ###
    train_df['age'] = train_df['age'].apply(age_map) ###

    test_df = test_df.merge(age_replace_df, on='location_city', how='left') ###
    test_df = test_df.drop(columns='age').rename(columns={'age_fillna':'age'}) ###
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean())) ###
    test_df['age'] = test_df['age'].apply(age_map) ###

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

def age_0413_ver4(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl=False) -> tuple:
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

    context_city = context_df[['location_city', 'age']].copy() ###
    context_city['location_city'] = context_city['location_city'].map(loc_city2idx) ###

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
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

def category_0414_ver1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 5개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    
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
    categoryhigh2idx = {v:k for k,v in enumerate(context_df['category_high'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
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
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def dohyun_0415_ver1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 10개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # year_of_publication 변수 전처리
    books.loc[104259, 'year_of_publication'] = 2010.0
    books.loc[121860, 'year_of_publication'] = 1997.0
    books = books.drop(np.where(books['year_of_publication'] < 1900)[0][0]).reset_index(drop=True)

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    # context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    # train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    # test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')


    train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    test_df['age'] = test_df['age'].apply(age_map)

    context_df = context_df.fillna('na') ; train_df = train_df.fillna('na') ; test_df = test_df.fillna('na')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(replace_na(context_df['location_city'].unique()))}
    loc_state2idx = {v:k for k,v in enumerate(replace_na(context_df['location_state'].unique()))}
    loc_country2idx = {v:k for k,v in enumerate(replace_na(context_df['location_country'].unique()))}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    # category2idx = {v:k for k,v in enumerate(replace_na(context_df['category'].unique()))}
    categoryhigh2idx = {v:k for k,v in enumerate(replace_na(context_df['category_high'].unique()))}
    publisher2idx = {v:k for k,v in enumerate(replace_na(context_df['publisher'].unique()))}
    language2idx = {v:k for k,v in enumerate(replace_na(context_df['language'].unique()))}
    author2idx = {v:k for k,v in enumerate(replace_na(context_df['book_author'].unique()))}

    # train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    # test_df['category'] = test_df['category'].map(category2idx)
    test_df['category_high'] = test_df['category_high'].map(categoryhigh2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        # "category2idx":category2idx,
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def dohyun_0415_ver3(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 10개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # year_of_publication 변수 전처리
    books.loc[104259, 'year_of_publication'] = 2010.0
    books.loc[121860, 'year_of_publication'] = 1997.0
    books = books.drop(np.where(books['year_of_publication'] < 1900)[0][0]).reset_index(drop=True)

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    # context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    # train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    # test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')


    train_df['age'] = train_df['age'].fillna(train_df['age'].median())
    test_df['age'] = test_df['age'].fillna(test_df['age'].median())

    context_df = context_df.fillna('na') ; train_df = train_df.fillna('na') ; test_df = test_df.fillna('na')
    
    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(replace_na(context_df['location_city'].unique()))}
    loc_state2idx = {v:k for k,v in enumerate(replace_na(context_df['location_state'].unique()))}
    loc_country2idx = {v:k for k,v in enumerate(replace_na(context_df['location_country'].unique()))}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    # category2idx = {v:k for k,v in enumerate(replace_na(context_df['category'].unique()))}
    categoryhigh2idx = {v:k for k,v in enumerate(replace_na(context_df['category_high'].unique()))}
    publisher2idx = {v:k for k,v in enumerate(replace_na(context_df['publisher'].unique()))}
    language2idx = {v:k for k,v in enumerate(replace_na(context_df['language'].unique()))}
    author2idx = {v:k for k,v in enumerate(replace_na(context_df['book_author'].unique()))}

    # train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    # test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        # "category2idx":category2idx,
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def dohyun_0415_ver4(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 10개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # year_of_publication 변수 전처리
    books.loc[104259, 'year_of_publication'] = 2010.0
    books.loc[121860, 'year_of_publication'] = 1997.0
    books = books.drop(np.where(books['year_of_publication'] < 1900)[0][0]).reset_index(drop=True)

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    test_df['age'] = test_df['age'].apply(age_map)

    context_df = context_df.fillna('na') ; train_df = train_df.fillna('na') ; test_df = test_df.fillna('na')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(replace_na(context_df['location_city'].unique()))}
    loc_state2idx = {v:k for k,v in enumerate(replace_na(context_df['location_state'].unique()))}
    loc_country2idx = {v:k for k,v in enumerate(replace_na(context_df['location_country'].unique()))}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(replace_na(context_df['category'].unique()))}
    categoryhigh2idx = {v:k for k,v in enumerate(replace_na(context_df['category_high'].unique()))}
    publisher2idx = {v:k for k,v in enumerate(replace_na(context_df['publisher'].unique()))}
    language2idx = {v:k for k,v in enumerate(replace_na(context_df['language'].unique()))}
    author2idx = {v:k for k,v in enumerate(replace_na(context_df['book_author'].unique()))}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['category_high'] = test_df['category_high'].map(categoryhigh2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def dohyun_0417_ver1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country', ascii=True):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city', ascii=True):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher', ascii=True):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing', ascii=True):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 10개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # year_of_publication 변수 전처리
    books.loc[104259, 'year_of_publication'] = 2010.0
    books.loc[121860, 'year_of_publication'] = 1997.0
    books = books.drop(np.where(books['year_of_publication'] < 1900)[0][0]).reset_index(drop=True)

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age_map'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    test_df['age_map'] = test_df['age'].apply(age_map)

    context_df = context_df.fillna('na') ; train_df = train_df.fillna('na') ; test_df = test_df.fillna('na')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(replace_na(context_df['location_city'].unique()))}
    loc_state2idx = {v:k for k,v in enumerate(replace_na(context_df['location_state'].unique()))}
    loc_country2idx = {v:k for k,v in enumerate(replace_na(context_df['location_country'].unique()))}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(replace_na(context_df['category'].unique()))}
    categoryhigh2idx = {v:k for k,v in enumerate(replace_na(context_df['category_high'].unique()))}
    publisher2idx = {v:k for k,v in enumerate(replace_na(context_df['publisher'].unique()))}
    language2idx = {v:k for k,v in enumerate(replace_na(context_df['language'].unique()))}
    author2idx = {v:k for k,v in enumerate(replace_na(context_df['book_author'].unique()))}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['category_high'] = test_df['category_high'].map(categoryhigh2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    # Feature Engineering
    category_high_dummies = [3,9,13,17,19,21,27,28,30,31,32,33,35,36,37,38,39,40]
    category_high_dummies = set(test_df['category_high'].unique()).intersection(category_high_dummies)
    category_high_dummies = ['category_high_' + str(x) for x in category_high_dummies]

    language_dummies = [2,4,5,6,7,8,9,11,12,13,14,16,17]
    language_dummies = set(test_df['language'].unique()).intersection(language_dummies)
    language_dummies = ['language_' + str(x) for x in language_dummies]

    train_category_high_new_features = pd.get_dummies(train_df['category_high'], prefix='category_high')[category_high_dummies]
    train_lang_new_features = pd.get_dummies(train_df['language'], prefix='language')[language_dummies]
    train_age_map_new_features = pd.get_dummies(train_df['age_map'], prefix='age_map')[['age_map_5']]

    test_category_high_new_features = pd.get_dummies(test_df['category_high'], prefix='category_high')[category_high_dummies]
    test_lang_new_features = pd.get_dummies(test_df['language'], prefix='language')[language_dummies]
    test_age_map_new_features = pd.get_dummies(test_df['age_map'], prefix='age_map')[['age_map_5']]

    train_df = pd.concat([train_df, train_category_high_new_features], axis=1)
    train_df = pd.concat([train_df, train_lang_new_features], axis=1)
    train_df = pd.concat([train_df, train_age_map_new_features], axis=1)

    test_df = pd.concat([test_df, test_category_high_new_features], axis=1)
    test_df = pd.concat([test_df, test_lang_new_features], axis=1)
    test_df = pd.concat([test_df, test_age_map_new_features], axis=1)

    return idx, train_df, test_df

def final(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country', ascii=True):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city', ascii=True):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher', ascii=True):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing', ascii=True):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 10개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<10]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # year_of_publication 변수 전처리
    books.loc[104259, 'year_of_publication'] = 2010.0
    books.loc[121860, 'year_of_publication'] = 1997.0
    books = books.drop(np.where(books['year_of_publication'] < 1900)[0][0]).reset_index(drop=True)

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    
    if is_dl:
        print('-'*20, 'Preprocess Done', '-'*20)
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'category_high', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    # age, year_of_publication mapping column -> feature로 적용
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age_map'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    test_df['age_map'] = test_df['age'].apply(age_map)

    context_df = context_df.fillna('na') ; train_df = train_df.fillna('na') ; test_df = test_df.fillna('na')
    
    yop_na_idx = test_df[test_df['year_of_publication']=='na'].index[0]
    test_df.loc[yop_na_idx,'year_of_publication'] = test_df.drop(index=yop_na_idx)['year_of_publication'].median()

    train_df['year_of_publication_map'] = train_df['year_of_publication'].apply(year_of_publication_map)
    test_df['year_of_publication_map'] = test_df['year_of_publication'].apply(year_of_publication_map)
    test_df['year_of_publication'] = test_df['year_of_publication'].astype(float)

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(replace_na(context_df['location_city'].unique()))}
    loc_state2idx = {v:k for k,v in enumerate(replace_na(context_df['location_state'].unique()))}
    loc_country2idx = {v:k for k,v in enumerate(replace_na(context_df['location_country'].unique()))}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(replace_na(context_df['category'].unique()))}
    categoryhigh2idx = {v:k for k,v in enumerate(replace_na(context_df['category_high'].unique()))}
    publisher2idx = {v:k for k,v in enumerate(replace_na(context_df['publisher'].unique()))}
    language2idx = {v:k for k,v in enumerate(replace_na(context_df['language'].unique()))}
    author2idx = {v:k for k,v in enumerate(replace_na(context_df['book_author'].unique()))}
    yopm2idx = {y:i for i, y in enumerate(pd.concat([train_df['year_of_publication_map'], test_df['year_of_publication_map']]).unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['category_high'] = train_df['category_high'].map(categoryhigh2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['year_of_publication_map'] = train_df['year_of_publication_map'].map(yopm2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['category_high'] = test_df['category_high'].map(categoryhigh2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['year_of_publication_map'] = test_df['year_of_publication_map'].map(yopm2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "categoryhigh2idx":categoryhigh2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "yopm2idx":yopm2idx
    }
    ### Feature Engineering

    # user별 평점의 [평균, 중앙값, 분산, 표준편차] feature 추가
    # FE_user = train_df[['user_id', 'rating']].groupby('user_id').aggregate([np.mean, np.median, np.var, np.std]).fillna(0)
    # FE_user = FE_user['rating'].rename(columns = {'mean':'mean_user', 'median':'median_user', 'var':'var_user', 'std':'std_user'})
    # train_df = train_df.merge(FE_user, how = 'left', left_on='user_id', right_on = 'user_id')
    # test_df = test_df.merge(FE_user, how = 'left', left_on='user_id', right_on = 'user_id')

    # category별 모든 유저의 평점의 [평균, 중앙값, 분산, 표준편차] feature 추가
    # idx2category = {v : k for k, v in category2idx.items()}
    # idx2categoryhigh = idx2category = {v : k for k, v in categoryhigh2idx.items()}

    # tmp_context_df = train_df.copy()
    # tmp_context_high_df = train_df.copy()
    # tmp_context_df.index = tmp_context_df.index.map(idx2category)
    # tmp_context_high_df.index = tmp_context_high_df.index.map(idx2categoryhigh)

    # FE_category = tmp_context_df.loc[:,['category', 'rating']].groupby('category').aggregate([np.mean, np.median, np.var, np.std])
    # FE_category = FE_category.fillna(FE_category.mean())
    # FE_category_high = tmp_context_high_df.loc[:,['category_high', 'rating']].groupby('category_high').aggregate([np.mean, np.median, np.var, np.std])
    # FE_category_high = FE_category_high.fillna(FE_category_high.mean())

    # for agg in ['mean', 'median', 'std']:
    #     train_df[f'category_{agg}'] = train_df['category'].map(FE_category.loc[:, 'rating'][agg])
    #     test_df[f'category_{agg}'] = test_df['category'].map(FE_category.loc[:, 'rating'][agg])
    #     train_df[f'category_high_{agg}'] = train_df['category_high'].map(FE_category_high.loc[:, 'rating'][agg])
    #     test_df[f'category_high_{agg}'] = test_df['category_high'].map(FE_category_high.loc[:, 'rating'][agg])

    # category별 각 유저의 평점의 [평균, 중앙값, 분산, 표준편차] feature 추가
    # FE_user_category = tmp_context_df.loc[:,['user_id', 'category', 'rating']].groupby(['user_id', 'category']).aggregate([np.mean, np.median, np.var, np.std])
    # FE_user_category = FE_user_category.fillna(FE_user_category.mean())
    # FE_user_category_high = tmp_context_df.loc[:,['user_id', 'category_high', 'rating']].groupby(['user_id', 'category_high']).aggregate([np.mean, np.median, np.var, np.std])
    # FE_user_category_high = FE_user_category_high.fillna(FE_user_category_high.mean())

    # train_df = train_df.merge(FE_user_category['rating'], how = 'left', left_on=['user_id', 'category'], right_on = ['user_id', 'category'])\
    #                 .merge(FE_user_category_high['rating'], how = 'left', left_on=['user_id', 'category_high'], right_on = ['user_id', 'category_high'], suffixes=('_user_category', '_user_category_high'))
    # test_df = test_df.merge(FE_user_category['rating'], how = 'left', left_on=['user_id', 'category'], right_on = ['user_id', 'category'])\
    #                 .merge(FE_user_category_high['rating'], how = 'left', left_on=['user_id', 'category_high'], right_on = ['user_id', 'category_high'], suffixes=('_user_category', '_user_category_high'))

    # rating이 5이면 1 아니면 0
    train_age_map_new_features = pd.get_dummies(train_df['age_map'], prefix='age_map')[['age_map_5']]
    test_age_map_new_features = pd.get_dummies(test_df['age_map'], prefix='age_map')[['age_map_5']]
    train_df = pd.concat([train_df, train_age_map_new_features], axis=1)
    test_df = pd.concat([test_df, test_age_map_new_features], axis=1)

    train_df['age_map_5'] = train_df['age_map_5'].map(lambda x : 1 if x else 0)
    test_df['age_map_5'] = test_df['age_map_5'].map(lambda x : 1 if x else 0)
    # del context_df, FE_user, tmp_context_df, tmp_context_high_df, FE_category, FE_category_high, FE_user_category, FE_user_category_high, train_age_map_new_features, test_age_map_new_features
    # del context_df, FE_user, tmp_context_df, tmp_context_high_df, FE_user_category, FE_user_category_high, train_age_map_new_features, test_age_map_new_features
    
    print('-'*20, 'Preprocess Done', '-'*20)

    return idx, train_df, test_df

def jisu_EDA_1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].map(lambda x : split_map(x)[0].strip()) # split : city, state, country
    users['location_state'] = users['location'].apply(lambda x : split_map(x)[1].strip())
    users['location_country'] = users['location'].apply(lambda x : split_map(x)[2].strip())
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4)'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            continue

    for location in tqdm(location_list, desc='(2/4)'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]

    users = users.drop(users[users['location_city'].isna() & users['location_country'].isna()].index) # country, city 둘 다 NaN인거 버리기

    country_list = users[users['location_city'].isna() & users['location_country'].notnull()]['location_country'].unique()

    for _country in tqdm(country_list, desc='(3/4)'):
        users[users['location']==_country] = users[users['location_country'] == 'usa'].groupby(by='location_city').count().sort_values('location_country', ascending=False).index[0]

    users = users.drop(users[users['location_city'].isna() | users['location_country'].isna()].index) # 나머지 null값 버리기


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(4/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 5개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # location은 이제 필요 없음
    users = users.drop(['location', 'location_state'], axis=1)
    print('-'*20, 'Preprocess Start', '-'*20)

    if is_dl:
        return users, books

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
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
        # "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def mission_1_EDA(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame, is_dl : bool = False) -> tuple:
    print('-'*20, 'Preprocess Start', '-'*20)
    # user preprocessing
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    # city는 있는데 country 없는 경우 채우기
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in tqdm(modify_location, desc='(1/4) fill country'):
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in tqdm(location_list, desc='(2/4) fill city'):
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


    # book preprocessing

    # 유명 출판사 표기 오류로 그룹화되지 못하는 케이스 처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in tqdm(modify_list, desc = '(3/4) grouping same publisher'):
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # category 대괄호 제거 및 소문자 변환
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # 43개의 high-category로 묶기
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
                  'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
                  'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
                  'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in tqdm(categories, desc = '(4/4) : high-categorizing'):
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # 5개 이하 항목 others로 묶기
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    # location은 이제 필요 없음
    users = users.drop(['location'], axis=1)
    print('-'*20, 'Preprocess Done', '-'*20)

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