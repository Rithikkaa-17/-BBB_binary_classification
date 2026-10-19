# Blood–Brain Barrier Permeability — Binary Classification

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
| **Q2** | Does the advantage depend on the learning algorithm? | Sign and significance consistent across two independently tuned tree ensembles |
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
```

`00_fetch_inputs.py` stages both source collections into `work/`: it copies the exact BBBP file
used for the published analysis from `data/BBBP_source.csv` (checksum-verified) and downloads
B3DB from the authors' repository. Pass `--b3db-path` to use a local copy on a machine without
network access. Every later script runs offline.

Checked on 19 October 2026 by re-running the pipeline from the source data: `00`–`01` rebuild the
external set at 2,201 molecules (1,100 permeable, 735 scaffold-overlapping, 1,466 scaffold-novel);
`02` retrains all six configurations over five seeds and reproduces every value in
`results/final_results.json` — each measure on both evaluation sets, every selected threshold and
both seed-summary conventions; `03` reproduces all thirty paired-bootstrap comparisons with their
intervals and p-values; and `10` reproduces the SRD ranking and the Pareto front. Note that B3DB is a living database: the
external set exactly as used is shipped at `data/external_B3DB_filtered.csv`.

## A note on reproducibility

Results are sensitive to library versions. Circular-fingerprint hashing and several descriptor
implementations change between RDKit releases, and tree construction changes between major
XGBoost versions. Re-running under the pinned environment in `requirements.txt` reproduces the
partition sizes and class balance exactly and the reported performance to within roughly 0.02
ROC-AUC. **The direction and significance of every comparison reported in the paper is unchanged
across the environments we tested.** Pin the versions before comparing numbers.
