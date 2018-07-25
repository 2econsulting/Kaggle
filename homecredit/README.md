### kaggle : homecredit 


|  ml | workflow | auc | time |
| ------------ | ------------ | ------------ |
| LGB | 기본 코드에서 결측값을 0으로 처리, max_model은 100 | 0.786 | 6hr |
| LGB | 기본 코드에서 결측값을 -9999으로 처리, max_model은 100 | 0.787 | 12hr |
| LGB | 기본 코드에서 결측값을 NA으로 처리, max_model은 100 | 0.786 | 6hr |