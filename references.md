# References

Key references for the DWC26 project. Each one is freely available online and describes a method applied in the final pipeline (`notebook2_proxy_model.ipynb`, `notebook3_hist_match.ipynb`, `notebook4_forecast.ipynb`, `src/`). The full working list, including methods that were tested but not adopted, is in `our_work/proxy_model_sijie/docs/references.md` and the Sources section of `summary_forecast.md`.

## 1. Organisers' reference paper

1. Zhang, Z., He, X., AlSinan, M., Kwak, H. and Hoteit, H. (2023). Robust method for reservoir simulation history matching using Bayesian inversion and long-short-term memory network-based proxy. *SPE Journal*, 28(03), 983–1007. https://doi.org/10.2118/203976-PA (open access, CC-BY 4.0; local copy in `reference/`)
   - Applied: the overall workflow of a neural-network proxy followed by Bayesian MCMC with the affine-invariant ensemble sampler, and joint tuning of the learning rate.

## 2. Proxy evaluation

2. Abdollahfard, Y., Fahimpour, J. and Ahmadi, M. (2025). Assessing proxy and AI models performance in waterflooding optimization. *Scientific Reports*, 15, 35472. https://doi.org/10.1038/s41598-025-19256-4 (open access)
   - Applied: proxies are judged on whether they reproduce the decision-relevant behaviour, not on R² alone. This is the basis of the ranking check in notebooks 2 and 3, which compares the history-matching score from the proxy with the score from the simulator.

## 3. Neural-network proxy

3. Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems 30*. arXiv:1612.01474. https://arxiv.org/abs/1612.01474
   - Applied: the proxy is an ensemble of 5 networks trained from different seeds (`src/proxy_lib.py`, `train_final`).
4. Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980. https://arxiv.org/abs/1412.6980
   - Applied: the optimiser for every proxy network.

## 4. MCMC history matching

5. Foreman-Mackey, D., Hogg, D. W., Lang, D. and Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306–312. arXiv:1202.3665. https://arxiv.org/abs/1202.3665
   - Applied: `emcee.EnsembleSampler` (the Goodman and Weare affine-invariant ensemble sampler) and the integrated autocorrelation time as a convergence check (`notebook3_hist_match.ipynb`).

## 5. Statistics

6. Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101. https://doi.org/10.2307/1412159 (public domain; free on JSTOR Early Journal Content)
   - Applied: Spearman rank correlation in the proxy ranking check and the forecast diagnostics (`scipy.stats.spearmanr`).

## 6. Production forecasting

7. Oliveira, Horowitz and Rojas Tueros (2021). CRM combined with the Koval and Gentil fractional-flow models ("Kogen"). *Oil & Gas Science and Technology*, 76. https://ogst.ifpenergiesnouvelles.fr/10.2516/ogst/2020090 (open access)
   - Applied: the adopted `crm_wor` forecast. A capacitance-resistance tank model gives the liquid rate from injection (time constant tau, gain G), and a fractional-flow split divides it into oil and water.
8. Oil & Gas Journal, "Technique forecasts production from waterflooded reservoirs". https://www.ogj.com/drilling-production/article/17255740/technique-forecasts-production-from-waterflooded-reservoirs
   - Applied: the water-oil-ratio split of `crm_wor`, a log WOR trend against cumulative oil.

## 7. Software documentation

| Library | Page | Applied for |
| --- | --- | --- |
| emcee | https://emcee.readthedocs.io/ | Ensemble sampler, autocorrelation analysis and convergence |
| PyTorch | `torch.optim.Adam`: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html | Proxy training |
