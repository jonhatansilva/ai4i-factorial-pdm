# When feature representation outweighs algorithm selection: a leakage-free factorial study of predictive maintenance on AI4I 2020

Code and results for a factorial, leakage-free evaluation of predictive-maintenance classifiers on the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i) (UCI).

The study asks how much of the performance comes from each modelling choice — validation protocol,
feature representation, class balancing and learning algorithm — and uses the fact that the AI4I failure
rules are known to test whether models recover the true failure mechanism.

## Main findings

- **Representation dominates in the deterministic failure modes.** For PWF and OSF, feature representation
  explains 33–67% of the PR-AUC variance across four ANOVA variants, more than the choice of algorithm.
- **The gain is not circular.** Features built automatically from all pairwise sums, differences, products
  and ratios of the sensors — with no domain knowledge — rediscover power (PWF), wear × torque (OSF) and the
  temperature difference (HDF), and almost match the physics-informed features.
- **Class balancing is redundant once the decision threshold is tuned.** No balancing technique improves
  PR-AUC, MCC or F1; SMOTE degrades the aggregated failure target.
- **Physics-informed features are more robust to missing data** than raw or automatic features.
- **Records are not independent.** Temperatures have lag-1 autocorrelation of 0.999 and all 115 HDF failures
  occur in a single episode; random splits inflate PR-AUC by about 0.10 without changing method rankings.
- With physics-informed features, depth-1/2 trees recover the generator rules (e.g. ΔT ≤ 8.7 K and
  rpm ≤ 1380 for HDF; power outside 3496–9001 W for PWF).

## Repository structure

| Path | Content |
|---|---|
| `notebook/desenho_fatorial_AI4I.ipynb` | Full pipeline (Experiments 1–8, sensitivity analyses), executed |
| `results/` | One CSV per experiment (raw results and summaries) |
| `figures/` | Publication figures (PNG 300 dpi and PDF) |
| `tables/tabelas_resultados.xlsx` | Result tables, one sheet per experiment |
| `scripts/figuras_finais.py` | Regenerates the figures from `results/` |

## Experiments

| Exp. | Question |
|---|---|
| 1 | Factorial design: protocol (temporal, random) × representation (raw, physics, automatic) × balancing (none, class weight, SMOTE) × model (logistic regression, shallow tree, random forest, gradient boosting), 4 splits per protocol, 5 targets (any failure, HDF, PWF, OSF, TWF) — 1,440 evaluations; ANOVA η² |
| 1.1 | Dependence between records (autocorrelation by record order) |
| 2 | Circularity: permutation importance in the automatic representation |
| 3 | Ablation of each physics-informed feature |
| 4 / 5 | Robustness to measurement noise / missing values |
| 6 | Hyper-parameter tuning vs. default configuration |
| 7 | Tree depth required to recover the failure rules |
| 8 | HDF with random splits (HDF never appears in temporal test sets) |
| S | Sensitivity: ANOVA without logistic regression and on the logit scale; paired Wilcoxon tests with Holm correction; balancing evaluated with MCC and F1 |

## Leakage control

Feature construction, imputation, scaling and SMOTE are fitted inside the pipeline on the training data only.
The decision threshold is chosen on a validation slice of the training data. Test sets keep the real class
distribution.

## How to run

```bash
git clone https://github.com/jonhatansilva/ai4i-factorial-pdm.git
cd ai4i-factorial-pdm
pip install -r requirements.txt
jupyter notebook notebook/desenho_fatorial_AI4I.ipynb
```

The notebook downloads the dataset from UCI if `manutencao_preditiva.csv` is not in the same folder.
Set `MODO_RAPIDO = True` for a test run of a few minutes. The full run takes about 1–3 hours on a laptop;
progress is checkpointed to CSV, so an interrupted run resumes where it stopped.

To regenerate the figures:

```bash
python scripts/figuras_finais.py results figures path/to/ai4i2020.csv
```

## Citation

Article in preparation. Please cite the dataset as: Matzka, S. (2020). Explainable artificial intelligence for
predictive maintenance applications. *Third International Conference on Artificial Intelligence for
Industries (AI4I)*, IEEE.

## License

MIT (see `LICENSE`).
