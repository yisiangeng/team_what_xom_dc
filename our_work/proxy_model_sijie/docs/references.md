# References

## Articles cited in the PRD

The researcher checks each "Cited in the PRD for" claim against the article before a spec cites it.

| Label | Cited in the PRD for | Full citation |
| --- | --- | --- |
| Article 2 | RNN/LSTM output layer, AIES sampler, cost metric | Zhang, Z., He, X., AlSinan, M., Kwak, H. and Hoteit, H. (2023). Robust method for reservoir simulation history matching using Bayesian inversion and long-short-term memory network-based proxy. *SPE Journal*, 28(03), 983–1007. https://doi.org/10.2118/203976-PA |
| Article 4 | Pooled R² (Eq. 16), relative observation error α (Eq. 18), selection filters | Thenon, A., Gervais, V. and Le Ravalec, M. (2022). Sequential design strategy for kriging and cokriging-based machine learning in the context of reservoir history-matching. *Computational Geosciences*, 26(5), 1101–1118. https://doi.org/10.1007/s10596-022-10147-5 |
| Article 6 | Mismatch rank ρ, local RMSE, σ_proxy, selection filters | Abdollahfard, Y., Fahimpour, J. and Ahmadi, M. (2025). Assessing proxy and AI models performance in waterflooding optimization. *Scientific Reports*, 15, 35472. https://doi.org/10.1038/s41598-025-19256-4 |

### Local PDF copies (repository root, not in git)

| File | Article | Text copy (read this) | Basis |
| --- | --- | --- | --- |
| `Robust Method for Reservoir Simulation History Matching Using Bayesian Inversion and Long-Short-Term Memory Network-Based Proxy.pdf` | Article 2 | `refs/article2.txt` (25 pages) | Page 1: title, authors, SPE Journal June 2023 p. 983 |
| `Sequential design strategy for kriging.pdf` | Article 4 | `refs/article4.txt` (18 pages) | Page 1: title, authors Thenon, Gervais, Le Ravalec |
| `evaluation method.pdf` | Article 6 | `refs/article6.txt` (36 pages) | Page 1: Heriot-Watt accepted manuscript of Abdollahfard et al. (2025), Scientific Reports 15, 35472 |

The text copies were extracted with `pypdf` on 13 Oct 2026, with `===== page N =====` markers. They are gitignored. Equations can lose their layout in the extraction, so check any equation cited in a spec against the PDF page.

**Page mapping.** Specs cite journal pages. The PDF page N of each text copy maps as follows:

- Article 2: journal page = N + 982. For example, PDF page 3 is p. 985.
- Article 4: journal page = N + 1100. For example, PDF page 3 is p. 1103.
- Article 6: article page = N − 1, because the accepted manuscript starts with a cover page. For example, PDF page 9 is p. 8.

### PRD claims checked (13 Oct 2026, Task 2 and Task 3 specs)

| Article | PRD claim | Finding | Used in |
| --- | --- | --- | --- |
| Article 2 | RNN/LSTM output: a linear layer with 3 units | Partly supported. The LSTM layers are combined with fully connected layers and a regression output layer (p. 984). The optimum has 2 LSTM layers (588 and 578 units), a 91-unit fully connected layer and a regression output (p. 991, Fig. 8, Table 3). Task 2 keeps the PRD's single linear layer, as an assumption | task2 §2.6 |
| Article 2 | (Input design, not a PRD claim) | 11 sequential inputs over 100 time steps (11 × 100), with 10 uncertainty parameters in Table 1. Thus parameters plus time at each step, which is option A (p. 984, p. 989). Inputs min-max normalised to (−1, 1) (p. 989). MSE objective (p. 985, Eq. 1). Bayesian optimisation of the learning rate, epochs, batch size and units (p. 985; ranges in Table 3, p. 991). Sensitivities (p. 994–995) | task2 §2.4, §2.6, §3.1 |
| Article 2 | Cost metric | Supported. One LSTM forward run takes about 0.1 s, and training (249 epochs) 33 min (p. 993) | task2 §2.15 |
| Article 2 | AIES sampler | Not yet checked (Task 4) | |
| Article 4 | Pooled R², Eq. 16 | **Partly supported (form only; re-checked for Task 3).** Eq. 16 (p. 1105) is R² = 1 − Σ_j (y(x̃_j) − ŷ(x̃_j))² ÷ Σ_j (y(x̃_j) − ȳ)² on an independent test sample, with ȳ the mean of the test outputs. Article 4 applies it to a scalar output, the objective function, on independent samples of 200 (Punq) and 101 (Brugge) points (p. 1114). It does not pool the dates of a time series | task3 §0.4, §2.2 (D-024); R²_O in §2.4 |
| Article 4 | (Not a PRD claim) | Matérn 5/2 covariance with one range for each parameter, and maximum-likelihood hyperparameters (p. 1103, Eq. 1–4; p. 1104, Eq. 5–7). Functional outputs: a POD basis truncated at a share of the explained variance, with one kriging model for each projection coefficient, built independently (p. 1106, Eq. 17) | task2 §2.9 |
| Article 4 | α, Eq. 18 | **Partly supported.** Eq. 18 (p. 1109) is OF(x) = ½ Σ_p ω_p Σ_k ((y_p(x, t_k) − y_ref,p(t_k)) ÷ σ_p(t_k))², with a generic σ_p(t). The 10% is in the case studies: Punq σ_p(t) = max(0.1 y_ref,p(t), 0.005) (p. 1111); Brugge "10% of the data, with a lower limit to 50" (p. 1113–1114). Both floors are absolute, unlike the PRD's β × max d. The PRD's O = 2 × OF with ω_p = 1 | task3 §0.4, §2.4 (D-025); Task 4 |
| Article 6 | (Not a PRD claim) | The GRU is a simplified LSTM with fewer parameters (p. 8). Latin hypercube designs (p. 7) | task2 §2.6 |
| Article 6 | Mismatch rank ρ, local RMSE | **Not supported as definitions.** Article 6 evaluates its proxies with RMSE (Eq. 21, p. 13), MAE, MAPE, R² (Eq. 24, p. 14) and relative error, on NPV. It has no rank correlation, no mismatch and no local error. It supports the motivation: proxies with test R² 0.90 to 0.97 (Table 8, p. 21) gave optimal points far from the simulator optimum (Euclidean distance, p. 16) and NPV errors ("Cal." against "Actual", Table 9, p. 30). It concludes that proxies must be judged on "optimization-relevant dynamics" (p. 1; p. 17–18) | task3 §0.4, §2.4: relabelled "assumption, motivated by Article 6" (D-025) |
| Article 6 | σ_proxy (PRD §8: "Assumption, after article 6") | Not in the article. The PRD already labels it an assumption | task3 §2.2; Task 4 |
| Article 6 | Selection filters (PRD §8, Task 4) | Not yet checked (Task 4) | |

## Project data sources

- **Profile (13 Oct 2026):** a read-only profile of `01 Introduction.xlsx` and `02 Field Production.xlsx` by the main session. It covers the sheet layout, sample ranges of X, the 11,096-row shared Date column, C5 passing for all cases, and the history rate matching the next-quarter mean (difference ≤ 3e-8) but not the previous-quarter mean (difference 1.0). Cited as "source: profile" in `docs/specs/task1.md`.

## Official references

- NumPy documentation, `numpy.interp`: https://numpy.org/doc/stable/reference/generated/numpy.interp.html (cited in Task 1 §2.4: piecewise-linear interpolant through (xp, fp). Bit-exactness at the knots is checked by test B08, not claimed by the documentation).

- Goodman, J. and Weare, J. (2010). Ensemble samplers with affine invariance. *Communications in Applied Mathematics and Computational Science*, 5(1), 65–80.
- Foreman-Mackey, D., Hogg, D. W., Lang, D. and Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306–312.
- emcee documentation, "Autocorrelation analysis and convergence": https://emcee.readthedocs.io/
- Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, 656, 5–28.
- Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L. and Kutz, J. N. (2014). On dynamic mode decomposition: Theory and applications. *Journal of Computational Dynamics*, 1(2), 391–421. https://doi.org/10.3934/jcd.2014.1.391. arXiv:1312.0041.
  - Task 2 cites Algorithm 2 (exact DMD). Algorithm 1 is standard DMD; the two differ in the modes, and share the eigenvalues of Ã.
  - Algorithm numbering checked via the arXiv record and a review summary on 13 Oct 2026, not against the typeset algorithm box.

### Task 2 (proxy training)

- Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. Algorithm 2.1 is the basis of scikit-learn's `GaussianProcessRegressor`. Chapter 4 covers covariance functions, including Matérn, and Chapter 5 covers marginal-likelihood hyperparameters.
- scikit-learn 1.7 documentation, `sklearn.gaussian_process.GaussianProcessRegressor`: https://scikit-learn.org/1.7/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html. It gives the constructor defaults (`alpha=1e-10`, `n_restarts_optimizer=0`, `normalize_y=False`), and states "based on Algorithm 2.1 of [RW2006]".
- scikit-learn 1.7.X source, `sklearn/gaussian_process/_gpr.py`: https://github.com/scikit-learn/scikit-learn/blob/1.7.X/sklearn/gaussian_process/_gpr.py. Checked 13 Oct 2026:
  - `predict` computes the variance from `kernel_.diag(X)`, so a `WhiteKernel` term is included.
  - `normalize_y` standardises y and maps the mean and variance back.
  - Restarts draw the initial θ uniformly within the log-bounds, from `random_state`.
- PyTorch 2.14 documentation, `torch.nn.LSTM`: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html. It gives the shapes (h₀ and c₀ are (num_layers, N, H) even with `batch_first`), the gate equations, and the default initialisation U(−√k, √k), k = 1/hidden_size. The `torch.nn.GRU` and `torch.nn.RNN` pages are analogous.
- PyTorch 2.14 documentation, "Reproducibility": https://docs.pytorch.org/docs/2.14/notes/randomness.html. It covers `torch.manual_seed`, `torch.use_deterministic_algorithms`, and "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
- optuna documentation, `optuna.samplers.TPESampler`: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html. The default `n_startup_trials` = 10, and random sampling is used until that many trials finish. Cited in D-015 as the reason not to add optuna.
- Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP 2014*, 1724–1734.
- Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv:1412.6980.
- Prechelt, L. (1998). Early stopping — but when? In G. B. Orr and K.-R. Müller (eds.), *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science 1524, 55–69. Springer.
- Pascanu, R., Mikolov, T. and Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning*, PMLR 28(3), 1310–1318.
- Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems 30*, 6402–6413.
- McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2), 239–245.
- Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.

### Task 2 amendment A2 (D-019, 13 Oct 2026)

Citation details were checked on 13 Oct 2026 against publisher and repository records. The article texts were not read; each is cited for its method only.

- Arps, J. J. (1945). Analysis of decline curves. *Transactions of the AIME*, 160(1), 228–247. https://doi.org/10.2118/945228-G. Cited for the exponential and hyperbolic decline forms (task2 §2.9a). The reading of the hyperbolic form as a positive gamma mixture of exponentials is our derivation, not Arps's claim.
- Golub, G. H. and Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. *SIAM Journal on Numerical Analysis*, 10(2), 413–432. https://doi.org/10.1137/0710036. Variable projection (task2 §2.9a step 4R).
- Chen, K. K., Tu, J. H. and Rowley, C. W. (2012). Variants of dynamic mode decomposition: boundary condition, Koopman, and Fourier analyses. *Journal of Nonlinear Science*, 22(6), 887–915. https://doi.org/10.1007/s00332-012-9130-9. Introduces optimized DMD, which fits the eigenvalues and amplitudes to all snapshots.
- Askham, T. and Kutz, J. N. (2018). Variable projection methods for an optimized dynamic mode decomposition. *SIAM Journal on Applied Dynamical Systems*, 17(1), 380–416. arXiv:1704.02343. Optimized DMD solved by variable projection. Task 2's real-eigenvalue constraint is our assumption, not a constraint from this article.
- SciPy documentation, `scipy.optimize.least_squares` (`method="trf"`, box bounds): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html. The pinned SciPy is 1.17.1 (V7).

### Task 3 (validation, backtest, selection, freeze and test)

Checked on 13 Oct 2026.

- Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101. The rank correlation of task3 §2.4 is the Pearson correlation of the ranks, with average ranks for ties.
- SciPy documentation, `scipy.stats.rankdata`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html. The default method is `average`: tied values get the mean of the ranks they would occupy.
- SciPy documentation, `scipy.stats.spearmanr`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html. Returns `statistic` and `pvalue`. Constant input gives NaN with a `ConstantInputWarning`. Test E05 compares with it.
- SQLite documentation, CREATE TRIGGER: https://www.sqlite.org/lang_createtrigger.html. `RAISE(ABORT, msg)` ends the statement and returns `SQLITE_CONSTRAINT` with the message. Non-TEMP triggers may query other tables of the same database. Used by the V3 triggers of task3 §1.6.
- Python 3.13 documentation, `sqlite3`: https://docs.python.org/3.13/library/sqlite3.html. `sqlite3.IntegrityError` is raised for constraint failures. That a trigger `RAISE(ABORT)` maps to it is checked by test E15, not stated in the documentation.
- Git documentation, `git merge-base --is-ancestor`: https://git-scm.com/docs/git-merge-base. Exit status 0 if the first commit is an ancestor of the second, 1 if not, another non-zero status on error. Used by guard G2 (task3 §1.10).
- Git documentation, `git ls-tree`: https://git-scm.com/docs/git-ls-tree. DESCRIPTION: the `<path>` is taken as relative to the current working directory, unless `--full-tree` is given. Gitglossary, pathspec: https://git-scm.com/docs/gitglossary. Pathspecs are used by `git diff` and `git ls-tree`, and the magic word `top` makes a pattern match from the root of the working tree "even when you are running the command from inside a subdirectory". Checked 14 Oct 2026. Basis of task3 §0.5 R1 (D-032).
- matplotlib 3.10.8 on PyPI: https://pypi.org/project/matplotlib/3.10.8/. Released 10 Dec 2025, Requires-Python ≥ 3.10, with a cp313 win_amd64 wheel (D-030).
- Arps (1945), above, is also the basis of the P3 threshold (D-022; task3 §2.7).
