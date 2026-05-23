

Saved simulated dataset to data/simulated_lactose_data.csv
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 CT   65                   0.00               1            2.59                   0
1                 CT   78                   9.20               1            7.80                   1
2                 TT   37                   8.74               0            4.02                   0
3                 CC   31                  15.87               0            8.94                   1
4                 TT   59                  11.03               0            4.75                   0

Class distribution:
lactose_intolerant
1    0.507
0    0.493
Name: proportion, dtype: float64
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 CT   65                   0.00               1            2.59                   0
1                 CT   78                   9.20               1            7.80                   1
2                 TT   37                   8.74               0            4.02                   0
3                 CC   31                  15.87               0            8.94                   1
4                 TT   59                  11.03               0            4.75                   0

               age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
count  1000.000000            1000.000000     1000.000000     1000.000000         1000.000000
mean     48.935000               8.121190        0.432000        5.242130            0.507000
std      17.890636               3.994266        0.495602        2.380542            0.500201
min      18.000000               0.000000        0.000000        0.000000            0.000000
25%      33.000000               5.417500        0.000000        3.290000            0.000000
50%      49.000000               8.165000        0.000000        5.065000            1.000000
75%      65.000000              10.755000        1.000000        7.192500            1.000000
max      79.000000              20.660000        1.000000       10.000000            1.000000

EDA plots saved to plots/

Lactose intolerance rate by genotype:
rs4988235_genotype
CC    0.822368
CT    0.282116
TT    0.136054
Name: lactose_intolerant, dtype: float64

Model comparison:
              feature_set                model  accuracy  precision    recall        f1
0  Without symptoms_score     Dummy Classifier     0.505   0.505000  1.000000  0.671096
1  Without symptoms_score  Logistic Regression     0.795   0.826087  0.752475  0.787565
2  Without symptoms_score        Decision Tree     0.650   0.663158  0.623762  0.642857
3  Without symptoms_score        Random Forest     0.755   0.770833  0.732673  0.751269
4     With symptoms_score     Dummy Classifier     0.505   0.505000  1.000000  0.671096
5     With symptoms_score  Logistic Regression     0.940   0.940594  0.940594  0.940594
6     With symptoms_score        Decision Tree     0.900   0.917526  0.881188  0.898990
7     With symptoms_score        Random Forest     0.925   0.938776  0.910891  0.924623

Saved model results to data/model_results.csv

Feature importance:
                      feature  importance
5         num__symptoms_score    0.722326
0  cat__rs4988235_genotype_CT    0.090009
3  num__dairy_intake_per_week    0.064327
1  cat__rs4988235_genotype_TT    0.058053
2                    num__age    0.050294
4         num__family_history    0.014991

Saved feature importance table to data/feature_importance.csv
Saved feature importance plot to plots/feature_importance.png
Saved confusion matrix to plots/confusion_matrix.png