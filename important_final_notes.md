
Made variants of synthetic code from the following research document - https://pubmed.ncbi.nlm.nih.gov/27577176/

Run the following to run the code: 

Remove-Item data\simulated_lactose_data.csv -ErrorAction SilentlyContinue
Remove-Item data\model_results.csv -ErrorAction SilentlyContinue
Remove-Item data\feature_importance.csv -ErrorAction SilentlyContinue

python src/generate_data.py
python src/exploratory_analysis.py
python src/train_models.py
python src/feature_importance.py
python src/confusion_matrix.py

