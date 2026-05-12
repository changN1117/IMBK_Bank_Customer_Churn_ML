#  IMBK_Bank_Customer_Churn_ML

## 1. 프로젝트명 
고객 이탈 분류 ML 및 인사이트 분석

## 2. 기간 
2026/04/10

## 3. 기술스택 
  - 데이터 전처리: Pandas, NumPy, Torch

  - 모델링/최적화: TabTransFormer

  - 시각화/해석: Matplotlib 

## 4. 데이터 출처 : 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

## 5. 데이터 전처리
   - 불필요한 피처 제거 : customer_id 
   - 범주형 데이터 인코딩 : country와 gender
   - 데이터 분할 : X = df.drop('churn', axis=1), y = df['churn']
                   train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["churn"])
   - 피처 스케일링 : StandardScaler

## 6. EDA 및 해석
고객들의 이탈률에는 상품의 보유 여부와 관련이 있을 것이라고 생각하여 ' 카드를 보유한 고객은 이탈률이 낮을 것이다.' 라는 가설을 들고 출발하였습니다. 그래서 신용카드 보유자와 신용카드 미보유자의 유지 및 이탈률을 확인을 해보았습니다.
     
<img width="580" height="455" alt="cardnotget" src="https://github.com/user-attachments/assets/a002335a-7d73-43a4-93ac-aee171c08d19" />
<img width="580" height="455" alt="cardget" src="https://github.com/user-attachments/assets/374b038f-2b1c-438f-affc-19529103d430" />

데이터 내에서 신용카드 보유자의 수는 7,055명이고 유지 고객은 5,631명, 이탈고객은 1,424명이었고 신용카드 미보유자 수는 2,945명에 유지고객은 2,332명이고  이탈고객은 613명이었습니다.
신용 카드의 보유 여부와는 관계 없이 이 두 집단의 이탈률을 25%와 26%로 차이는 미미 했으며 검증 결과 신용카드의 보유 여부는 고객의 이탈률과는 관계가 없다는 결론을 내렸습니다.
   

## 7. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

AutoML을 이용해 f1_score가 가장 높게 나온 모델 4개를 찾았습니다.
<img width="742" height="555" alt="best_model" src="https://github.com/user-attachments/assets/b7d66538-05ae-4d91-acf4-9e640a46a8e7" />

그 후 개별 모델의 f1_score를 극대화하기 위한 최적의 파라미터 조합을 탐색하기 위하여 optuna를 통해 각 모델당 10회 이상의 반복시행을 하여 최적값을 도출했습니다.

<img width="353" height="98" alt="옵튜나png" src="https://github.com/user-attachments/assets/e9cfac6a-a301-4524-a684-440f6491f1f9" />

그렇게 나온 최적의 4가지 모델들을 결합하여 스태킹 모델을 만들었습니다. 
<img width="583" height="296" alt="스태킹 모델" src="https://github.com/user-attachments/assets/a1fd212c-ed5f-496c-97d3-935b4d129301" />

성능을 향상시키기 위해서 4개의 모델을 결합하여 학습을 진행하였지만 뚜렷한 성능의 향상은 일어나지 않았습니다.
<img width="660" height="100" alt="스택킹" src="https://github.com/user-attachments/assets/240ce33f-7641-4f34-bb7c-801460580f7c" />

가장 성능이 뛰어난 cat boost모델로 shap value를 시각화 했습니다.
<img width="766" height="557" alt="shap-value" src="https://github.com/user-attachments/assets/3523ac0b-115b-4bba-b022-0471b0206fda" />

도출된 shap value의 해석본입니다.
- 이용 상품 수가 많을수록 고객이 유지되고 적을수록 이탈할 가능성이 높습니다 하지만 너무 많은 상품의 이용은 고객이 이탈할 가능성이 있습니다
- 나이가 어릴수록 유지될 확률이 높고 높읈록 이탈할 확률이 높습니다
- 활성고객이 활성화 될수록 유지 확률이 높고 낮을수록 이탈 확률이 높습니다
- 계좌 잔액이 적을수록 유지 확률이 높고 높을수록 이탈 확률이 높습니다
- 남성일수록 유지확률이 높고 여성일수록 이탈확률이 높습니다
- 프랑스에 사는 사람일수록 유지확률이 높고 스페인에 살수록 이탈확률이 높습니다
- 추정연봉에 대해서는 너무 섞여있어서 파악이 어렵습니다
- 신용점수가 높을수록 유지확률이 높고 낮을수록 이탈확률이 높습니다.
- 이용기간이 높을수록 유지확률이 높고 적을수록 이탈확률이 높습니다
- 신용카드를 보유하지 않은 고객은 이탈할 확률이 있지만 그렇게 큰 차이는 없어보입니다.

## 8. 인사이트 제안
도출된 shap value의 결과 남성 고객은 유지를 하는 반면 여성 고객의 이탈을 많이 하는 모습을 보이고 나이가 많을수록 이탈률이 높아지는 모습을 보이고 있었습니다.
그래서 노년층 여성 고객의 이탈을 막기 위해서 노년 여성을 위한 상품을 준비하는 것이 좋아보였습니다. 또한 노년 대책 상품을 마련한다면 남성들의 유지율 역시 더 올라갈 것이라고 생각하여 은퇴 후 대책을  마련하는데 도움이 되는 상품을 개발하는 것이 좋아보인다고 생각했습니다. 
