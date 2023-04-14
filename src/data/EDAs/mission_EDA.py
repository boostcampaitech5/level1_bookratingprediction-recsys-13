import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def mission_1_EDA(users : pd.DataFrame, books : pd.DataFrame) -> tuple:
    print('-'*20, 'Mission1 EDA Start', '-'*20)
    # user preprocessing
    users['location_city'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
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
    print('-'*20, 'Mission1 EDA Done', '-'*20)
    return users, books