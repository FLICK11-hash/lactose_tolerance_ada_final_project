
# Predicting Lactose Intolerance from Simulated Genetic and Symptom Data (Changed a bit for Final checkpoint)

This project builds a machine learning pipeline that uses simulated genetic, demographic, family history, dairy intake, and symptom-based features to predict lactose intolerance status.
The goal is not to create a medical diagnostic tool. Instead, the project studies how genetic and symptom-based features can be used in a predictive model, 
how different models compare, and what limitations exist when using simplified biological data for prediction.
The dataset is simulated because public individual-level genetics datasets related to lactose intolerance can be limited, complex, or difficult to use responsibly. 
The simulated data includes features such as rs4988235 genotype, age, dairy intake per week, family history, and symptom score.
The project compares several models, including a majority-class baseline, Logistic Regression, Decision Tree, and Random Forest. 
Model performance is evaluated using accuracy, precision, recall, and F1-score. The project also analyzes feature importance to understand which variables contribute most to prediction.
A key limitation is that the model may rely heavily on symptom score, which is closely related to the target label. This means the model may be learning symptom patterns more than purely genetic risk. 
The project also discusses ethical concerns, including population bias, incomplete biological information, and the danger of treating a simplified model as a medical diagnosis.

Execution Plan By Week 5 Homework Deadline:
Identify one or more public genetics-related datasets, or design a simulated dataset if the public data is too limited or too complex. 
Define the target variable clearly as whether an individual is likely lactose intolerant or lactose tolerant. Clean the dataset and perform exploratory data analysis. 
Research the biological meaning of the available variables, especially the genetic markers most commonly associated with lactose persistence and lactose intolerance. 
Build baseline models and compare them to a simple baseline such as majority-class prediction.

By Week 7 Homework Deadline:
Train more advanced models and compare them against the baseline models.
Evaluate performance using appropriate metrics such as accuracy.
Analyze which genetic features appear most predictive and whether the models remain interpretable.
Investigate overfitting and test whether regularization or feature selection improves generalization.
Begin documenting the practical and ethical limitations of the project, including bias, human errors, incomplete biological information, and misuse risks.

By the Final Due Date:
Finalize the predictive pipeline and comparison across models.
Create clear visualizations for feature importance, performance, and uncertainty.
Write up the main findings, especially focusing on how well lactose intolerance can be predicted and what the model cannot reliably conclude.
Record a final video demonstrating the pipeline, explaining the code, and discussing the scientific and ethical tradeoffs of predicting lactose intolerance from genetic data.
