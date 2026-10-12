Bayesian ROC experiments

main.tex is the reference for the scientific definitions and protocols.
The current manuscript uses synthetic protocol v8 and real-data protocol v3.
Both completed runs and their generated tables are available.

Current experiments

Both protocols evaluate DiP-ROC predictions for a future empirical ROC and AUC
on a fixed grid, with the same class counts as the observed test sample.
They use 4,000 predictive draws and nominal levels of 80% and 95%.

Synthetic (v8):
- Easy, Medium and Hard score scenarios; definitions are in
  roc_utils/simulation_datasets.py.
- Five positive/negative count pairs: (20,20), (100,100), (500,500),
  (20,500) and (500,20); 10,000 replications per configuration; 10 bins.
- Nine settings: uniform, historical-AUC linear and historical-histogram
  priors, each with total concentration m=1, 5 or 20 per class.
- Each replication has independent test, external and historical samples.
  The informed priors share 500 historical observations per class.
  History determines the prior profile; m controls its weight. Historical
  observations are not added to the test sample.

Real data (v3):
- Twelve datasets, defined in roc_utils/real_datasets.py; 300 splits each.
- Tables report nine datasets, excluding Alzheimer, Fertility and Sonar;
  validation still covers the complete twelve-dataset source CSV.
- Test and external samples each receive floor(class_count/5) observations
  per class; the remaining observations are used for training.
- Preprocessing and logistic regression are fitted on training data only.
  All grids reuse the same fitted model and held-out scores.
- Uniform prior with m=1; 10 bins for the main results, 5 and 20 for grid checks.
  Missing source files are downloaded and checked by the dataset loader.

Run experiments and generate tables

Run from py/ with the project environment activated:
  source .venv/bin/activate

Synthetic:
  python run_experiment_main_evaluation_on_synthetic_data.py
  python run_tab_experiment_main_evaluation_on_synthetic_data.py

Real data:
  python run_experiment_empirical_evaluation_on_real_world_datasets.py
  python run_tab_experiment_empirical_evaluation_on_real_world_datasets.py

Experiment commands run the full budgets above and refuse existing CSVs by
default. To regenerate tables from an existing completed run, use only its
table command. Table generators validate the CSV and its metadata, write
LaTeX fragments, and do not edit main.tex.

Outputs, relative to py/:
- results/current_experiments/main_evaluation_on_synthetic_data/
  raw_replications_v8.csv and latex_tables_v8/
- results/current_experiments/empirical_evaluation_on_real_world_datasets/
  raw_replications_v3.csv and latex_tables_v3/

Keep each CSV with its checksum-named .metadata.json companion. It records
the configuration, source hashes, software and random streams. Both protocols
use PCG64 with default seed 20250731. Do not mix results from different
protocol versions or overwrite earlier results and archives.

Small checks and tests

For a reduced run, import run_experiment from the relevant runner and pass
n_replications=2, n_samples=40 and a separate output_csv. For pilot tables,
pass that CSV as input_csv to generate_latex_tables, explicitly set a separate
output_dir, and pass the same budgets and any selected conditions.
Pilot results must not replace manuscript results or official tables.

Run the complete test suite from the repository root:
  BAYESIAN_ROC_NETWORK_TESTS=0 py/.venv/bin/python -m unittest discover -s py/tests -p 'test_*.py' -v

For Python code changes, add or update precise tests and run the full suite.
Do not use python -O: runtime assertions are part of the experiment checks.

Other scripts

run_simulation_score_distributions.py regenerates the synthetic score figures.
The sensitivity_to_prior_specification scripts are separate exploratory
experiments, outside the current manuscript protocol.
Keep back_up_legacy_pipeline/ and results/archived_experiments/ unchanged
unless explicitly requested.
