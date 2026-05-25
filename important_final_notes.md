# Final Notes

The synthetic data generation was revised using ideas from the following research document:

https://pubmed.ncbi.nlm.nih.gov/27577176/

The project uses simulated genotype probabilities and risk assumptions inspired by published research, but the dataset is not real patient data. The purpose of the synthetic dataset is to test the machine learning pipeline and study how different feature choices affect prediction.

One important revision was changing how `dairy_intake_per_week` affects the model. 
Dairy intake no longer directly increases the probability of lactose intolerance, because eating dairy does not cause lactose intolerance. 
Instead, dairy intake affects how visible symptoms are. This is more realistic because people who rarely eat dairy, such as vegans or people who avoid dairy, 
may not show strong symptoms even if they are lactose intolerant.

The symptom score is now adjusted by dairy exposure. 
If someone consumes little or no dairy, their observed `symptoms_score` may be lower because they have not had enough exposure for symptoms to appear. 
If someone consumes more dairy, symptoms are more likely to be visible.

## How to Run the Code

Remove-Item data\simulated_lactose_data.csv -ErrorAction SilentlyContinue
Remove-Item data\model_results.csv -ErrorAction SilentlyContinue
Remove-Item data\feature_importance.csv -ErrorAction SilentlyContinue

python src/generate_data.py
python src/exploratory_analysis.py
python src/train_models.py
python src/feature_importance.py
python src/confusion_matrix.py

