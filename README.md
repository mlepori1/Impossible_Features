# Impossible_Features

## Reproducing Experimental Results

### Study 1 + 2:
First, run `fit_pca.py` using the configs found in `src/configs/Utils_Fit_PCA`. This will fit PCA objects to the last-token hidden states of all models using the Wikitext validation set.

Next, create modal difference vectors (and identify best PCs + Random Vectors) using `exp1_create_representations.py` and the configs found in `src/configs/Exp1_Create_Representations`.

Use these modal difference vectors (and other baselines) to classify stimuli using `exp1_classify.py` and the configs found in `src/configs/Exp1_Classify`.

Analyze these results to produce the figures from Study 1 and Study 2 using `analysis/analyze_exp1.ipynb`.

### Study 3:
Use these modal difference vectors (and other baselines) to featurize stimuli using `exp3_calibration.py` and the configs found in `src/configs/Exp3_Calibration`.

Fit QDA models to produce the figures from Study 3 using `analysis/analyze_exp3.ipynb`.

### Study 4:
Get projections along  modal difference vectors using `exp4_correlate.py` and the configs found in `src/configs/Exp4_Correlation`.

Correlate these projections with known features using `analysis/analyze_exp4.ipynb`.