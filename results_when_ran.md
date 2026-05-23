Saved simulated dataset to data/simulated_lactose_data.csv
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 TT   71                   6.83               0            2.56                   0
1                 CT   63                   6.07               1            4.21                   1
2                 CC   39                   0.16               1            0.06                   0
3                 CC   27                   3.33               1            2.32                   1
4                 CC   32                   4.35               0            4.15                   1

Class distribution:
lactose_intolerant
1    0.539
0    0.461
Name: proportion, dtype: float64
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 TT   71                   6.83               0            2.56                   0
1                 CT   63                   6.07               1            4.21                   1
2                 CC   39                   0.16               1            0.06                   0
3                 CC   27                   3.33               1            2.32                   1
4                 CC   32                   4.35               0            4.15                   1

               age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
count  1000.000000            1000.000000     1000.000000     1000.000000         1000.000000
mean     49.714000               8.041720        0.465000        4.357110            0.539000
std      18.200535               3.929263        0.499023        2.595137            0.498726
min      18.000000               0.000000        0.000000        0.000000            0.000000
25%      34.000000               5.235000        0.000000        2.350000            0.000000
50%      50.000000               8.020000        0.000000        3.955000            1.000000
75%      66.000000              10.747500        1.000000        6.462500            1.000000
max      79.000000              21.270000        1.000000       10.000000            1.000000

EDA plots saved to plots/

Lactose intolerance rate by genotype:
rs4988235_genotype
CC    0.834395
CT    0.310345
TT    0.190789
Name: lactose_intolerant, dtype: float64

Model comparison:
              feature_set                model  accuracy  precision    recall        f1
0  Without symptoms_score     Dummy Classifier     0.540   0.540000  1.000000  0.701299
1  Without symptoms_score  Logistic Regression     0.805   0.863158  0.759259  0.807882
2  Without symptoms_score        Decision Tree     0.710   0.745098  0.703704  0.723810
3  Without symptoms_score        Random Forest     0.730   0.759615  0.731481  0.745283
4     With symptoms_score     Dummy Classifier     0.540   0.540000  1.000000  0.701299
5     With symptoms_score  Logistic Regression     0.955   0.962617  0.953704  0.958140
6     With symptoms_score        Decision Tree     0.860   0.857143  0.888889  0.872727
7     With symptoms_score        Random Forest     0.920   0.942308  0.907407  0.924528

Saved model results to data/model_results.csv

Feature importance:
                      feature  importance
5         num__symptoms_score    0.499792
3  num__dairy_intake_per_week    0.213925
0  cat__rs4988235_genotype_CT    0.108662
2                    num__age    0.081539
1  cat__rs4988235_genotype_TT    0.075488
4         num__family_history    0.020594

Saved feature importance table to data/feature_importance.csv
Saved feature importance plot to plots/feature_importance.png
Saved confusion matrix to plots/confusion_matrix.png