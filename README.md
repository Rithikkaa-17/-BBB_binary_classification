# Blood–Brain Barrier Permeability - Binary Classification

Code and data for the study *Representation Outweighs Algorithm in Blood–Brain Barrier
Permeability Prediction: A Factorial Study with Multi-Criteria Model Selection and
Scaffold-Stratified External Validation*.

Everything reported in the manuscript can be regenerated from this repository.

---

## What the project asks

Most published models for this endpoint report a single number from a single algorithm on a
single held-out split. This study was built around four questions instead, each with a decision
criterion fixed in advance:

| | Question | Criterion |
|---|---|---|
| **Q1** | Do structural fingerprints and computed physicochemical descriptors carry *complementary* information? | Paired bootstrap interval for the fusion difference excludes zero, tested within each algorithm separately |
| **Q2** | Does the advantage depend on the learning algorithm? | Sign and significance consistent across three learners at fixed settings, and across a sweep of feature subsampling |
| **Q3** | How much performance survives a move to unseen chemistry? | External collection sharing no molecular skeleton with training data, stratified by scaffold novelty |
| **Q4** | Can a 248-molecule test set rank competing models? | Whether paired tests on it separate configurations that a larger external set separates |

## Headline results:

- **Fusion wins within both algorithms.** Concatenating ten descriptors onto a 2,048-bit
  fingerprint improves external ranking performance in both the random forest and the
  gradient-boosted ensemble, with all four confidence intervals excluding zero.
- **Algorithm choice does not decide it.** The random forest with fused features reaches an
  external ROC-AUC of **0.781** as a mean over five seeds (**0.782** for the seed-averaged
  predictor); the gradient-boosted equivalent reaches 0.754. The two are *not* separable on the
  internal test set.

> **Two reporting conventions.** Every configuration is refitted under five seeds. A *per-seed
> mean ± SD* scores each seed separately and averages the five scores; a *seed-averaged* figure
> scores the mean of the five seed models' predicted probabilities, which is the final predictor
> and admits no standard deviation. The two differ in the third decimal place. `results/
> per_seed_results.json` holds both, together with all thirty underlying per-seed values and the
> validation-selected threshold for every seed, so any figure in the paper can be traced to the
> convention it was computed under.
- **The internal test set cannot rank models.** Most pairwise differences on 248 molecules have
  confidence intervals spanning zero — e.g. 0.032 with an interval of −0.028 to 0.092 (p = 0.30).
- **External performance is lower than internal, and stratification matters.** 0.854 on compounds
  whose scaffold occurs in the development set against **0.740** on scaffold-novel compounds.
- **Probabilities do not transfer.** Well calibrated internally (Brier 0.091, expected
  calibration error 0.070) but over-confident externally (Brier 0.236, expected calibration
  error 0.199), because training prevalence (0.77)
  differs from external prevalence (0.50). The model is a ranking filter, not a calibrated
  probability estimator.

## Added at revision

Reviewer comments on the BioMedInformatics submission prompted seven further analyses. Three
claims in the submitted manuscript did not survive them and were withdrawn.

**The error set is not "intermediate polarity".** The submitted text described the misclassified
molecules as lying where the descriptors discriminate least. Checked against all 34 errors,
that is wrong: they span polar surface area 33.1 to 215.0, the widest range in the test
set, with median 79.8 against 45.2 for correct predictions, and only 47% fall
inside the interquartile band of the correct predictions against 50% of all test
molecules. What the errors are instead: 29 false positives at median polar surface area
73 and 5 false negatives at median 148 — opposite ends, not a middle band — and
22 of 34 fall on scaffolds carrying more than one error, with two congeneric research
series accounting for 17 between them. Both series carry *both* labels on one Bemis–Murcko
scaffold, so the model scores the series rather than the molecule. See
`results/error_analysis.json` and `results/error_clustering.json`.

**The fingerprint block carries more total attribution than the descriptors.** The submitted text
claimed the model relies predominantly on interpretable bulk properties rather than idiosyncratic
fingerprint bits. Summed over blocks, the ten descriptors carry 26.2% of total mean
|SHAP| and the 2,048 bits the remaining 73.8%. What survives is the per-feature statement: a
descriptor carries about 73 times the attribution of an average bit, and six descriptors
outrank every individual bit. See `results/block_attribution.json`.

**Version sensitivity was overstated.** The submitted §2.11 claimed differences "of order 0.02"
between software environments. Rebuilt end to end under rdkit 2026.09.1 / scikit-learn 1.9.1 /
xgboost 3.4.1, the headline model moves by 0.0006 and the largest shift across all six
configurations is 0.0019 — an order of magnitude below the fusion effect. All four fusion
comparisons remain separable in the second environment. See `results/environment_A.json` and
`results/environment_B.json`.

**What held up.** The fusion effect survives every check applied to it:

| Check | Result |
|---|---|
| `max_features` swept over a factor of 56 | fusion beats fingerprint at every setting, all intervals exclude zero (+0.0222 to +0.0269) |
| gradient-boosting `colsample_bytree` 0.3 / 0.7 / 1.0 | separable at all three |
| SVM with Tanimoto kernel (no feature subsampling at all) | +0.0228 [+0.0144, +0.0311] |
| second software environment | all four comparisons separable |
| desalted, neutralised structures | separable; standardisation *improves* external ROC-AUC by +0.0045 [+0.0022, +0.0068] |

At the scikit-learn default of `max_features='sqrt'` a split considers 45 of 2,058 columns, so
the probability it can see any descriptor is 0.199. The sweep exists because that number
makes the default a plausible confounder; it is not one.

**Other additions.** Target-population recalibration recovers most of the calibration deficit
(expected calibration error 0.199 → 0.046 with Platt scaling, 0.029 with isotonic
regression; `results/recalibration.json`). Crossing source grouping with scaffold novelty shows
the novelty penalty is itself source-dependent — group A falls 0.876 → 0.744 and group B
0.966 → 0.728, while group C does not fall (0.708 → 0.731);
`results/group_x_novelty.json`. The highest-attribution fingerprint bits are mapped through
`bitInfo` to the environments that set them, and collisions are present, so no chemical meaning
is assigned to an individual bit (`results/bit_mapping_and_dependence.json`). Attribution against
logP is strongly non-monotone — linear R² 0.40, quadratic R² 0.89 — so a Pearson
coefficient understates it.

## Added at revision

Reviewer comments on the BioMedInformatics submission prompted seven further analyses. Three
claims in the submitted manuscript did not survive them and were withdrawn.

**The error set is not "intermediate polarity".** The submitted text described the misclassified
molecules as lying where the descriptors discriminate least. Checked against all 34 errors,
that is wrong: they span polar surface area 33.1 to 215.0, the widest range in the test
set, with median 79.8 against 45.2 for correct predictions, and only 47% fall
inside the interquartile band of the correct predictions against 50% of all test
molecules. What the errors are instead: 29 false positives at median polar surface area
73 and 5 false negatives at median 148 — opposite ends, not a middle band — and
22 of 34 fall on scaffolds carrying more than one error, with two congeneric research
series accounting for 17 between them. Both series carry *both* labels on one Bemis–Murcko
scaffold, so the model scores the series rather than the molecule. See
`results/error_analysis.json` and `results/error_clustering.json`.

**The fingerprint block carries more total attribution than the descriptors.** The submitted text
claimed the model relies predominantly on interpretable bulk properties rather than idiosyncratic
fingerprint bits. Summed over blocks, the ten descriptors carry 26.2% of total mean
|SHAP| and the 2,048 bits the remaining 73.8%. What survives is the per-feature statement: a
descriptor carries about 73 times the attribution of an average bit, and six descriptors
outrank every individual bit. See `results/block_attribution.json`.

**Version sensitivity was overstated.** The submitted §2.11 claimed differences "of order 0.02"
between software environments. Rebuilt end to end under rdkit 2026.09.1 / scikit-learn 1.9.1 /
xgboost 3.4.1, the headline model moves by 0.0006 and the largest shift across all six
configurations is 0.0019 — an order of magnitude below the fusion effect. All four fusion
comparisons remain separable in the second environment. See `results/environment_A.json` and
`results/environment_B.json`.

**What held up.** The fusion effect survives every check applied to it:

| Check | Result |
|---|---|
| `max_features` swept over a factor of 56 | fusion beats fingerprint at every setting, all intervals exclude zero (+0.0222 to +0.0269) |
| gradient-boosting `colsample_bytree` 0.3 / 0.7 / 1.0 | separable at all three |
| SVM with Tanimoto kernel (no feature subsampling at all) | +0.0228 [+0.0144, +0.0311] |
| second software environment | all four comparisons separable |
| desalted, neutralised structures | separable; standardisation *improves* external ROC-AUC by +0.0045 [+0.0022, +0.0068] |

At the scikit-learn default of `max_features='sqrt'` a split considers 45 of 2,058 columns, so
the probability it can see any descriptor is 0.199. The sweep exists because that number
makes the default a plausible confounder; it is not one.

**Other additions.** Target-population recalibration recovers most of the calibration deficit
(expected calibration error 0.199 → 0.046 with Platt scaling, 0.029 with isotonic
regression; `results/recalibration.json`). Crossing source grouping with scaffold novelty shows
the novelty penalty is itself source-dependent — group A falls 0.876 → 0.744 and group B
0.966 → 0.728, while group C does not fall (0.708 → 0.731);
`results/group_x_novelty.json`. The highest-attribution fingerprint bits are mapped through
`bitInfo` to the environments that set them, and collisions are present, so no chemical meaning
is assigned to an individual bit (`results/bit_mapping_and_dependence.json`). Attribution against
logP is strongly non-monotone — linear R² 0.40, quadratic R² 0.89 — so a Pearson
coefficient understates it.

## Added at revision

Reviewer comments on the BioMedInformatics submission prompted seven further analyses. Three
claims in the submitted manuscript did not survive them and were withdrawn.

**The error set is not "intermediate polarity".** The submitted text described the misclassified
molecules as lying where the descriptors discriminate least. Checked against all 34 errors,
that is wrong: they span polar surface area 33.1 to 215.0, the widest range in the test
set, with median 79.8 against 45.2 for correct predictions, and only 47% fall
inside the interquartile band of the correct predictions against 50% of all test
molecules. What the errors are instead: 29 false positives at median polar surface area
73 and 5 false negatives at median 148 — opposite ends, not a middle band — and
22 of 34 fall on scaffolds carrying more than one error, with two congeneric research
series accounting for 17 between them. Both series carry *both* labels on one Bemis–Murcko
scaffold, so the model scores the series rather than the molecule. See
`results/error_analysis.json` and `results/error_clustering.json`.

**The fingerprint block carries more total attribution than the descriptors.** The submitted text
claimed the model relies predominantly on interpretable bulk properties rather than idiosyncratic
fingerprint bits. Summed over blocks, the ten descriptors carry 26.2% of total mean
|SHAP| and the 2,048 bits the remaining 73.8%. What survives is the per-feature statement: a
descriptor carries about 73 times the attribution of an average bit, and six descriptors
outrank every individual bit. See `results/block_attribution.json`.

**Version sensitivity was overstated.** The submitted §2.11 claimed differences "of order 0.02"
between software environments. Rebuilt end to end under rdkit 2026.09.1 / scikit-learn 1.9.1 /
xgboost 3.4.1, the headline model moves by 0.0006 and the largest shift across all six
configurations is 0.0019 — an order of magnitude below the fusion effect. All four fusion
comparisons remain separable in the second environment. See `results/environment_A.json` and
`results/environment_B.json`.

**What held up.** The fusion effect survives every check applied to it:

| Check | Result |
|---|---|
| `max_features` swept over a factor of 56 | fusion beats fingerprint at every setting, all intervals exclude zero (+0.0222 to +0.0269) |
| gradient-boosting `colsample_bytree` 0.3 / 0.7 / 1.0 | separable at all three |
| SVM with Tanimoto kernel (no feature subsampling at all) | +0.0228 [+0.0144, +0.0311] |
| second software environment | all four comparisons separable |
| desalted, neutralised structures | separable; standardisation *improves* external ROC-AUC by +0.0045 [+0.0022, +0.0068] |

At the scikit-learn default of `max_features='sqrt'` a split considers 45 of 2,058 columns, so
the probability it can see any descriptor is 0.199. The sweep exists because that number
makes the default a plausible confounder; it is not one.

**Other additions.** Target-population recalibration recovers most of the calibration deficit
(expected calibration error 0.199 → 0.046 with Platt scaling, 0.029 with isotonic
regression; `results/recalibration.json`). Crossing source grouping with scaffold novelty shows
the novelty penalty is itself source-dependent — group A falls 0.876 → 0.744 and group B
0.966 → 0.728, while group C does not fall (0.708 → 0.731);
`results/group_x_novelty.json`. The highest-attribution fingerprint bits are mapped through
`bitInfo` to the environments that set them, and collisions are present, so no chemical meaning
is assigned to an individual bit (`results/bit_mapping_and_dependence.json`). Attribution against
logP is strongly non-monotone — linear R² 0.40, quadratic R² 0.89 — so a Pearson
coefficient understates it.

## Added at revision

Reviewer comments on the BioMedInformatics submission prompted seven further analyses. Three
claims in the submitted manuscript did not survive them and were withdrawn.

**The error set is not "intermediate polarity".** The submitted text described the misclassified
molecules as lying where the descriptors discriminate least. Checked against all 34 errors,
that is wrong: they span polar surface area 33.1 to 215.0, the widest range in the test
set, with median 79.8 against 45.2 for correct predictions, and only 47% fall
inside the interquartile band of the correct predictions against 50% of all test
molecules. What the errors are instead: 29 false positives at median polar surface area
73 and 5 false negatives at median 148 — opposite ends, not a middle band — and
22 of 34 fall on scaffolds carrying more than one error, with two congeneric research
series accounting for 17 between them. Both series carry *both* labels on one Bemis–Murcko
scaffold, so the model scores the series rather than the molecule. See
`results/error_analysis.json` and `results/error_clustering.json`.

**The fingerprint block carries more total attribution than the descriptors.** The submitted text
claimed the model relies predominantly on interpretable bulk properties rather than idiosyncratic
fingerprint bits. Summed over blocks, the ten descriptors carry 26.2% of total mean
|SHAP| and the 2,048 bits the remaining 73.8%. What survives is the per-feature statement: a
descriptor carries about 73 times the attribution of an average bit, and six descriptors
outrank every individual bit. See `results/block_attribution.json`.

**Version sensitivity was overstated.** The submitted §2.11 claimed differences "of order 0.02"
between software environments. Rebuilt end to end under rdkit 2026.09.1 / scikit-learn 1.9.1 /
xgboost 3.4.1, the headline model moves by 0.0006 and the largest shift across all six
configurations is 0.0019 — an order of magnitude below the fusion effect. All four fusion
comparisons remain separable in the second environment. See `results/environment_A.json` and
`results/environment_B.json`.

**What held up.** The fusion effect survives every check applied to it:

| Check | Result |
|---|---|
| `max_features` swept over a factor of 56 | fusion beats fingerprint at every setting, all intervals exclude zero (+0.0222 to +0.0269) |
| gradient-boosting `colsample_bytree` 0.3 / 0.7 / 1.0 | separable at all three |
| SVM with Tanimoto kernel (no feature subsampling at all) | +0.0228 [+0.0144, +0.0311] |
| second software environment | all four comparisons separable |
| desalted, neutralised structures | separable; standardisation *improves* external ROC-AUC by +0.0045 [+0.0022, +0.0068] |

At the scikit-learn default of `max_features='sqrt'` a split considers 45 of 2,058 columns, so
the probability it can see any descriptor is 0.199. The sweep exists because that number
makes the default a plausible confounder; it is not one.

**Other additions.** Target-population recalibration recovers most of the calibration deficit
(expected calibration error 0.199 → 0.046 with Platt scaling, 0.029 with isotonic
regression; `results/recalibration.json`). Crossing source grouping with scaffold novelty shows
the novelty penalty is itself source-dependent — group A falls 0.876 → 0.744 and group B
0.966 → 0.728, while group C does not fall (0.708 → 0.731);
`results/group_x_novelty.json`. The highest-attribution fingerprint bits are mapped through
`bitInfo` to the environments that set them, and collisions are present, so no chemical meaning
is assigned to an individual bit (`results/bit_mapping_and_dependence.json`). Attribution against
logP is strongly non-monotone — linear R² 0.40, quadratic R² 0.89 — so a Pearson
coefficient understates it.

## Data

| Split | Molecules | Purpose |
|---|---|---|
| Training | 1,632 | model fitting |
| Validation | 159 | hyperparameters, decision threshold |
| Development | 1,791 | training + validation; final refit |
| Internal test | 248 | scored once |
| External | 2,201 | independent collection |

Partitioning is by Bemis–Murcko scaffold, so no scaffold spans two partitions.

**Sources.** The training collection is the BBBP set distributed with MoleculeNet. The external
set derives from [B3DB](https://github.com/theochem/B3DB) (Meng et al., *Sci. Data* 2021).

**External set construction.** Each structure is reduced to the 14-character connectivity layer of
its InChIKey — invariant to salt form, stereochemistry and isotopic substitution. Skeletons with
conflicting labels inside B3DB are dropped; duplicates are collapsed; every skeleton occurring
anywhere in the source collection is removed. This leaves 2,201 compounds.

> **Important distinction.** Skeleton deduplication is *not* scaffold deduplication. A
> Bemis–Murcko scaffold is a substructure shared by many distinct molecules, so removing identical
> skeletons leaves scaffold overlap behind. Of the 2,201 external compounds, 735 carry a scaffold
> that also occurs in the development set and 1,466 do not. The two notions are reported
> separately throughout; conflating them inflates apparent external performance.

## Repository layout

```
data/      source collection, partition assignments (CSV) and held-out probabilities (NPZ)
src/       numbered pipeline scripts, run in order
results/   published JSON outputs quoted in the manuscript
figures/   publication figures at 600 dpi, numbered as in the manuscript
work/      created on first run; intermediate artefacts, git-ignored
```

All paths in `src/` resolve relative to the repository root, so the scripts run from
anywhere once the repository is cloned.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/00_fetch_inputs.py              # stage BBBP and B3DB into work/
python src/01_curate_and_split.py          # curate, scaffold-split, build external set
python src/02_train_and_evaluate.py        # six configurations x five seeds
python src/03_statistical_comparison.py    # paired bootstrap comparisons
python src/04_revision_analyses.py         # Tanimoto, strata, class weighting
python src/05_attribution_and_calibration.py
python src/06_make_figures.py
python src/07_molecules_and_beeswarm.py
python src/08_per_seed_results.py        # per-seed and seed-averaged ROC, thresholds
python src/09_source_strata.py           # external performance by source grouping
python src/10_multicriteria.py           # SRD and Pareto at the selected thresholds

# added at revision (BioMedInformatics, reviewers 1 and 2)
python src/11_subsampling_and_learners.py   # max_features / colsample sweep + SVM-Tanimoto
python src/12_error_analysis.py             # what the errors actually are
python src/13_block_attribution.py          # attribution by feature block, all five seeds
python src/14_standardisation_ablation.py   # desalt / neutralise + ionisation descriptors
python src/15_recalibration.py              # Platt and isotonic on target-population data
python src/16_bit_mapping_and_dependence.py # bitInfo mapping, Spearman, curvature
python src/17_second_environment.py         # same comparison under a different toolchain
python src/18_figure_s1_dependence.py       # Figure S1
```

`00_fetch_inputs.py` stages both source collections into `work/`: it copies the exact BBBP file
used for the published analysis from `data/BBBP_source.csv` (checksum-verified) and downloads
B3DB from the authors' repository. Pass `--b3db-path` to use a local copy on a machine without
network access. Every later script runs offline.

Re-checked on 7 November 2026 by re-running the pipeline from the source data on a different machine, in the pinned environment. The result is reported as a tolerance rather than as exact reproduction, and the two kinds of measure behave differently.

`00`–`01` rebuild the external set exactly: 2,201 molecules, 1,100 permeable, 735 scaffold-overlapping, 1,466 scaffold-novel, with the rebuilt CSV identical row for row to the shipped `data/external_B3DB_filtered.csv`.

**Threshold-free measures reproduce.** Across the 36 ROC-AUC, PR-AUC and Brier values in `results/final_results.json`, the largest difference is 0.0019 and the median is 0.00014; none exceeds 0.01. Seed-mean ROC-AUC moves by at most 0.00245. All 30 paired-bootstrap comparisons reproduce to within 0.0033 in the difference and every separability verdict is unchanged. The Pareto front is the same single configuration.

**Threshold-dependent measures do not, and the reason is informative.** Across the 72 accuracy, balanced-accuracy, precision, recall, F1 and MCC values the largest difference is 0.0472 (RF + Fused, internal MCC: 0.4287 against 0.3816) and 7 exceed 0.01. The cause is not the models but the operating point: the threshold is chosen by maximising MCC on 159 validation molecules, and that arg-max is unstable. Probability shifts of order 1e-4 move the selected threshold by up to 0.07, which then moves MCC on the test set by up to 0.047. 3 of the six configurations select an identical threshold on re-execution; 3 do not.

This is the same phenomenon the manuscript reports for model selection on 248 molecules, applied to threshold selection on 159, and it is worth stating plainly: every primary comparison in the study is threshold-free and is unaffected, but the threshold-dependent columns of the performance tables should be read as having a reproducibility tolerance of several hundredths, not several ten-thousandths. An earlier version of this file claimed exact reproduction; that was true of the original machine and is not guaranteed on another, and the claim has been corrected.

B3DB is a living database: the external set exactly as used is shipped at `data/external_B3DB_filtered.csv`.

## A note on reproducibility

Results are sensitive to library versions. Circular-fingerprint hashing and several descriptor
implementations change between RDKit releases, and tree construction changes between major
XGBoost versions. Re-running under the pinned environment in `requirements.txt` reproduces the
partition sizes and class balance exactly and the reported performance to within roughly 0.02
ROC-AUC. **The direction and significance of every comparison reported in the paper is unchanged
across the environments we tested.** Pin the versions before comparing numbers.
