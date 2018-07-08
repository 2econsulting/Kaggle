## Accuracy of each file 

|  file name | accuracy | threshold |
| ------------ | ------------ | ------------ |
|  PYTHON_TEST_LIGHTGBM_TUNE_CVPRED.csv | 0.79904 | >0.5 |
|  PYTHON_TEST_LIGHTGBM_DEFAULT_CVPRED.csv | 0.77511 | >0.5 |
| PYTHON_TEST_XGBOOST_TUNE_CVPRED | 0.77511 | >0.5 |
| PYTHON_TEST_XGBOOST_DEFAULT_CVPRED | 0.78468 | >0.5 |
| PYTHON_TEST_CATBOOST_TUNE_CVPRED | 0.78947 | >0.5 |
| PYTHON_TEST_CATBOOST_DEFAULT_CVPRED | 0.79425 | >0.5 |
|  R_TEST_H2OLGB_DEFAULT_CVPRED.csv | 0.77033  | >0.5 |
|  R_TEST_H2OXGB_DEFAULT_CVPRED.csv | 0.76076  | >0.5 |
|  R_TEST_CATBOOST_DEFAULT_CVPRED.csv | 0.79904  | >0.5 |
|  R_TEST_CATBOOST_TUNE_P1.csv | 0.82296 | >0.5 |
|  R_TEST_CATBOOST_TUNE_CVP1.csv | 0.81818 | >0.5 |
|  R_TEST_H2OXGB_TUNE_CVP1.csv | 0.81339 | >0.5 |
|  R_TEST_CATBOOST_TUNE_CVP1_78.csv | 0.81339 | >0.5 |
|  R_TEST_H2OLGB_TUNE_CVP1.csv | 0.81818 | >0.5 |  
|  R_TEST_XGB_TUNE_CVPRED.csv | 0.78468 | >0.5 | 


### 예측력 0.82775 달성

* 사용된 파일(+알고리즘)
  * R_TEST_CATBOOST_TUNE_P1.csv
  * R_TEST_CATBOOST_TUNE_CVP1.csv
  * PYTHON_TEST_LIGHTGBM_TUNE_CVPRED.csv
* 절차
  * 위 파일들에 각각 확률 값을 불러온다.
  * 각 확률을 0.5 기준으로 1 아니면 0으로 바꾼다.
  * 바꾼 0,1 칼럼을 행별로 평균을 구한다
  * 구한 평균을 0.5 기준으로 1 아니면 0으로 바꾼다.
  * 위 값을 submission