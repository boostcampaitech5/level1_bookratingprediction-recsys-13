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

def dohyun_0417_ver1(users : pd.DataFrame, books : pd.DataFrame, ratings1 : pd.DataFrame, ratings2 : pd.DataFrame) -> tuple:
    print('-'*20, 'Mission1 EDA Start', '-'*20)
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
    print('-'*20, 'Mission1 EDA Done', '-'*20)

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