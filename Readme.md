## 🚀프로젝트 개요
![image](https://user-images.githubusercontent.com/72483874/233927053-ae2a8f87-a402-45a2-8bf9-2a33753c7b29.png)
### ⭐️프로젝트 주제

- 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 태스크입니다.
- 해당 경진대회는 이러한 소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회입니다.
- 리더보드는 평점 예측에서 자주 사용되는 지표 중 하나인 RMSE (Root Mean Square Error)를 사용합니다.

### 💻활용 장비 및 재료

- ai stage server : V100 GPU x 4
- python==3.8.5
- torch==1.7.1
- CUDA==11.0

### 🗂️프로젝트 구조 및 사용 데이터셋의 구조도

📁**level1_bookratingprediction-recsys-13** <br>
┣ 📝ensemble.py <br>
┣ 📝main.py <br>
┣ 📁src <br>
ㅣ┣ 📁preprocess <br>
ㅣ┣ 📁ensembles <br>
ㅣ┣ 📁models <br>
ㅣ┗ 📁 train <br>
┗📁submit <br>

📁data<br>
**┣**📝users.csv<br>
┣📝books.csv<br>
┣📝sample_submission.csv<br>
┣📝test_ratings.csv<br>
┗ 📝train_ratings.csv<br>

- requirements : `install requirements`

```
  pip install -r requirements.txt
```

- train & Inference : `main.py`

```
  python main.py --help
  python main.py --MODEL catboost --DATA_PATH /opt/ml/data
```

- Ensemble the four inferences
```
  python ensenmble.py <filenames> <weights>
```

## 🤝프로젝트 팀 구성 및 역할

&nbsp;

<table align="center">
  <tr height="155px">
    <td align="center" width="150px">
      <a href="https://github.com/HeewonKwak"><img src="https://avatars.githubusercontent.com/HeewonKwak"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/limstonestone"><img src="https://avatars.githubusercontent.com/limstonestone"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/zsmalla"><img src="https://avatars.githubusercontent.com/zsmalla"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/hoyajigi"><img src="https://avatars.githubusercontent.com/hoyajigi"/></a>
    </td>
  </tr>
  <tr height="80px">
    <td align="center" width="150px">
      <a href="https://github.com/HeewonKwak">곽희원_T5015</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/limstonestone">임도현_T5170</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/zsmalla">임지수_T5176</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/hoyajigi">조현석_T5205</a>
    </td>
  </tr>
</table>
&nbsp;

- 곽희원_T5015
    - 프로젝트 개발(`EDA`, `Hyper Parameter Tuning`, `Stratified K-Fold`, `Modeling`, `Ensemble`)
- 임도현_T5170
    - 프로젝트 개발(`EDA`,  `Data Preprocessing`, `Feature Engineering`, `Modeling`)
- 임지수_T5176
    - 프로젝트 개발(`EDA`, `Data Preprocessing`, `Ensemble`)
- 조현석_T5205
    - 프로젝트 개발(`EDA`, `Hyper Parameter Tuning`, `Manage Data Pipeline`)

## 📅프로젝트 수행 절차 및 방법

- 수행 절차

![image](https://user-images.githubusercontent.com/72483874/233926654-e130c4f0-707e-4387-862b-731ebd07875c.png)

- 수행 방법

![image](https://user-images.githubusercontent.com/72483874/233926736-16e09481-0336-448d-bba6-5dc81b39d34d.png)

## 🏆프로젝트 수행 결과
  - 모델 성능
      - CatBoost 모델을 사용하여 예측하여, 테스트 데이터셋에 대한 예측을 진행하였습니다.예측 결과 RMSE는 2.1181로, 대회의 6위 성적을 기록하였습니다.
            
          ![image](https://user-images.githubusercontent.com/72483874/233929089-d2060ab6-f15a-46d5-9c60-2fab3ac6559d.png)
          ![image](https://user-images.githubusercontent.com/72483874/233929238-706896c5-5cd2-42b9-ad34-0f32294332d0.png)
<h3>
  <p align="center">
    <a href="WrapUpReport-Recsys13.pdf">[📑 Wrap Up Report]</a>
  </p>
</h3>

            
