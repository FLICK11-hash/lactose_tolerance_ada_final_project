(.venv) PS C:\Users\conra\lactose_tolerance_ada_final_project> python src/generate_data.py
>> python src/exploratory_analysis.py
>> python src/train_models.py
>> python src/feature_importance.py
>> python src/confusion_matrix.py
>> ls data
>> ls plots
Saved simulated dataset to data/simulated_lactose_data.csv
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 CC   64                  11.04               0            7.42                   1
1                 TT   29                   9.12               1            3.87                   0
2                 CT   79                   8.42               1            1.48                   0
3                 CT   33                   7.75               0            6.98                   1
4                 CC   41                   4.98               0            5.78                   1

Class distribution:
lactose_intolerant
1    0.516
0    0.484
Name: proportion, dtype: float64
  rs4988235_genotype  age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
0                 CC   64                  11.04               0            7.42                   1
1                 TT   29                   9.12               1            3.87                   0
2                 CT   79                   8.42               1            1.48                   0
3                 CT   33                   7.75               0            6.98                   1
4                 CC   41                   4.98               0            5.78                   1

               age  dairy_intake_per_week  family_history  symptoms_score  lactose_intolerant
count  1000.000000            1000.000000     1000.000000     1000.000000         1000.000000
mean     48.381000               8.137500        0.459000        5.033600            0.516000
std      17.817612               3.836105        0.498566        2.353795            0.499994
min      18.000000               0.000000        0.000000        0.000000            0.000000
25%      32.000000               5.475000        0.000000        3.027500            0.000000
50%      48.000000               8.050000        0.000000        4.900000            1.000000
75%      64.000000              10.657500        1.000000        7.012500            1.000000
max      79.000000              20.770000        1.000000       10.000000            1.000000

EDA plots saved to plots/

Lactose intolerance rate by genotype:
rs4988235_genotype
CC    0.843818
CT    0.272021
TT    0.143791
Name: lactose_intolerant, dtype: float64

Model comparison:
                 model  accuracy  precision    recall        f1
0     Dummy Classifier     0.515   0.515000  1.000000  0.679868
1  Logistic Regression     0.955   0.951923  0.961165  0.956522
2        Decision Tree     0.930   0.923810  0.941748  0.932692
3        Random Forest     0.955   0.951923  0.961165  0.956522

Saved model results to data/model_results.csv

Feature importance:
                      feature  importance
5         num__symptoms_score    0.681978
0  cat__rs4988235_genotype_CT    0.098484
3  num__dairy_intake_per_week    0.076235
1  cat__rs4988235_genotype_TT    0.069919
2                    num__age    0.060464
4         num__family_history    0.012920

Saved feature importance table to data/feature_importance.csv
Saved feature importance plot to plots/feature_importance.png
Saved confusion matrix to plots/confusion_matrix.png


    Directory: C:\Users\conra\lactose_tolerance_ada_final_project\data


Mode                 LastWriteTime         Length Name                                                                       
----                 -------------         ------ ----                                                                       
-a----          5/6/2026   1:10 PM            273 feature_importance.csv                                                     
-a----          5/6/2026   1:10 PM            327 model_results.csv                                                          
-a----          5/6/2026   1:10 PM          21198 simulated_lactose_data.csv                                                 


    Directory: C:\Users\conra\lactose_tolerance_ada_final_project\plots


Mode                 LastWriteTime         Length Name                                                                       
----                 -------------         ------ ----                                                                       
-a----          5/6/2026   1:10 PM          12673 class_distribution.png                                                     
-a----          5/6/2026   1:10 PM          19623 confusion_matrix.png                                                       
-a----          5/6/2026   1:10 PM          26523 feature_importance.png                                                     
-a----          5/6/2026   1:10 PM          11086 genotype_distribution.png                                                  
-a----          5/6/2026   1:10 PM          16667 intolerance_rate_by_genotype.png                                           
-a----          5/6/2026   1:10 PM          17351 symptoms_by_genotype.png
