# Robust Method for Reservoir Simulation History Matching Using Bayesian Inversion and Long-Short-Term Memory Network-Based Proxy

**Zhen Zhang¹, Xupeng He¹, Marwah AlSinan², Hyung Kwak², and Hussein Hoteit¹\***

¹ King Abdullah University of Science and Technology (KAUST)
² Saudi Aramco

\*Corresponding author; email: hussein.hoteit@kaust.edu.sa

*SPE Journal, June 2023, pp. 983–1007. SPE-203976-PA.*
Copyright © 2023 The Authors. Published by the Society of Petroleum Engineers. Published under a Creative Commons Attribution License (CC-BY 4.0).
This paper (SPE 203976) was accepted for presentation at the SPE Reservoir Simulation Conference, On-Demand, 26 October 2021, and revised for publication. Original manuscript received for review 9 July 2022. Revised manuscript received for review 12 October 2022. Paper peer approved 17 October 2022.

---

## Summary

History matching is a critical process used for calibrating simulation models and assessing subsurface uncertainties. This common technique aims to align the reservoir models with the observed data. However, achieving this goal is often challenging due to the nonuniqueness of the solution, underlying subsurface uncertainties, and usually the high computational cost of simulations. The traditional approach is often based on trial and error, which is exhaustive and labor-intensive. Some analytical and numerical proxies combined with Monte Carlo simulations are used to reduce the computational time. However, these approaches suffer from low accuracy and may not fully capture subsurface uncertainties.

This study proposes a new robust method using Bayesian Markov chain Monte Carlo (MCMC) to perform assisted history matching under uncertainties. We propose a novel three-step workflow that includes (1) multiresolution low-fidelity models to guarantee high-quality matching; (2) long-short-term memory (LSTM) network as a low-fidelity model to reproduce continuous time response based on the simulation model, combined with Bayesian optimization to obtain the optimum low-fidelity model; and (3) Bayesian MCMC runs to obtain the Bayesian inversion of the uncertainty parameters.

We perform sensitivity analysis on the LSTM's architecture, hyperparameters, training set, number of chains, and chain length to obtain the optimum setup for Bayesian-LSTM history matching. We also compare the performance of predicting the recovery factor (RF) using different surrogate methods, including polynomial chaos expansions (PCE), kriging, and support vector machines for regression (SVR). We demonstrate the proposed method using a water flooding problem for the upper Tarbert formation of the 10th SPE comparative model. This study case represents a highly heterogeneous nearshore environment.

Results showed that the Bayesian-optimized LSTM has successfully captured the physics in the high-fidelity model. The Bayesian-LSTM MCMC produces an accurate prediction with narrow ranges of uncertainties. The posterior prediction through the high-fidelity model ensures the robustness and accuracy of the workflow. This approach provides an efficient and practical history-matching method for reservoir simulation and subsurface flow modeling with significant uncertainties.

---

## Introduction

Accurate modeling of multiphase fluid flow in subsurface reservoirs is critical in a wide range of environmental and energy applications, including geothermal extraction (Hoteit et al. 2021; Santoso et al. 2019a, 2019b; He et al. 2021a), oil recovery in fractured reservoirs (He et al. 2021b, 2022a), and underground carbon storage (Hoteit et al. 2019; Pal et al. 2022). The process requires history matching and model calibration to minimize the mismatch between predictions and observed data (AlAmeri et al. 2020; Omar et al. 2021; Santoso et al. 2019a, 2019b; Ţene et al. 2016). To improve model predictability, common procedures often involve adjusting geological characteristics (such as porosity and permeability), fluid properties (e.g., fluid viscosity and density), and rock-fluid interactions (e.g., capillary pressure and relative permeability) (AlAmeri et al. 2020; Liu and Durlofsky 2020).

History matching remains challenging and sometimes misleading due to the possibility of generating multiple feasible numerical solutions due to subsurface uncertainties and the need for expensive computations to perform high-resolution simulations (Liu and Durlofsky 2020; Tang et al. 2020; Katterbauer et al. 2015). The nonuniqueness of the solution is often caused by the limited amount of observed data, uncertainties, and the high dimensionality of the model. History matching is, therefore, an ill-posed inverse problem that carries various uncertainties (Elsheikh et al. 2012). The costly computations result from the necessity to run the forward model numerous times to determine the disparity between simulated and observed data. Thus, there is a need for a robust history-matching workflow that:

1. reduces the cost of performing forward runs without decreasing the matching accuracy;
2. efficiently accommodates the uncertainties during the calibration process; and
3. preserves the underlying physics and geology.

Traditional history-matching techniques often rely on multipliers to adjust the parameters to minimize the mismatch, where a trial-and-error process is carried out until a specific level of precision is achieved (Agarwal et al. 2000; Williams et al. 1998). This approach often lacks rigor and robustness due to a large number of degrees of freedom and the subjectivity of the process, which makes it inefficient and labor-intensive (Awemo et al. 2014; Oliver and Chen 2011). Other advanced history-matching methods use automated procedures to update the matching parameters in the model (Oliver and Chen 2011; Rwechungura et al. 2011). Various algorithms have been proposed in the literature for history matching, including evolutionary algorithm, that is, genetic algorithm; evolutionary strategy (Oliver and Chen 2011; Romero and Carter 2001; Schulze-Riegert et al. 2002, 2003); gradual deformation, that is, sequential Gaussian simulation, truncated Gaussian, multipoint statistics (Caers 2003; Caers and Hoffman 2006; Feraille et al. 2003; Hoffman and Caers 2004; Hu 2000; Hu et al. 2001; Zhang and Oliver 2011); neighborhood algorithm (Oliver and Chen 2011; Suzuki et al. 2008); adjoint-based sensitivity (Chavent et al. 1975; Chen et al. 1974); streamline-based sensitivity (Agarwal and Blunt 2003); gradient methods, that is, Gauss-Newton, Levenberg-Marquardt, conjugate gradient, Broyden-Fletcher-Goldfarb-Shanno (Lee et al. 1986; Makhlouf et al. 1993; Vefring et al. 2006; Yang and Watson 1988; Zhang et al. 2003), randomized maximum likelihood (Kitanidis 1995; Oliver et al. 1996), Bayesian MCMC (Elsheikh et al. 2012; Oliver et al. 1996), and Kalman filter (Haugen et al. 2008; Zhang and Oliver 2011). Bayesian MCMC has recently gained popularity as a history-matching method for its rigorous and efficient framework to account for uncertainties during the matching process (Elsheikh et al. 2012; Hamdi et al. 2017; Marzouk et al. 2007; Martino 2018).

In this work, we propose a new Bayesian framework that is insensitive to different prior distributions, does not need expensive computations, and can represent continuous time response using LSTM. We use multifidelity models with parameter interactions to ensure the robustness and accuracy of the obtained history matching. The LSTM model is trained to replace the high-fidelity physical model and reduce the computational cost. One can use different approaches for the design of experiments (DoEs) to produce the training samples, such as the Box-Behnken design, the central-composite design, and the Latin hypercube design. To ensure space filling, we adopted the Latin hypercube design, as suggested by Li et al. (2019). Moreover, the affine invariant ensemble (AIES) algorithm is deployed to perform the Bayesian MCMC, where the LSTM model serves as the forward model. Finally, the mean value from the posterior distribution is fed into the high-fidelity physical model to ensure accurate matching and consistency of the physical parameters. To our knowledge, this workflow is applied in reservoir simulation for the first time.

We demonstrate the workflow for waterflooding in a 3D heterogeneous oil reservoir with a five-spot pattern, corresponding to the upper Tarbert formation of the tenth SPE comparative model (Christie and Blunt 2001). The workflow provides the prediction of time-dependent RF and posterior distributions of more than 10 uncertainty parameters, including residual saturations, relative permeabilities, permeability and porosity scaling factor, and injection rate. The inferred pairwise scatter plot identifies correlation among the parameters.

Bayesian MCMC uses prior distributions to capture the posterior distributions through statistical samplings, such as Metropolis-Hastings, reversible jump, and Hamiltonian Monte Carlo (Zhang and Curtis 2020). Each realization is followed by a forward run of the physical model, which generates a response that is used to propose the next realization. The posterior distribution (i.e., the final product) inherits uncertainties and preserves the physics of the model (Marzouk et al. 2007). However, there are several drawbacks in the process, including (1) expensive computational cost of the simulations as the Bayesian MCMC often requires thousands of forward runs to converge (Elsheikh et al. 2012; Marzouk et al. 2007) and (2) the history-matching accuracy depends heavily on proper prior information (Jagalur-Mohan et al. 2018; Marzouk et al. 2007; Winkler 1967).

To reduce the computational cost, many authors (Marzouk et al. 2007; Elsheikh et al. 2012; Hamdi et al. 2017; Naik et al. 2019) considered low-fidelity (surrogate) models to represent the high-fidelity physical model. Marzouk et al. (2007) used PCE for a diffusive transport case. Elsheikh et al. (2012) used Gaussian process regression for a waterflooding study case. Hamdi et al. (2017) deployed a kriging metamodel for the assessment of unconventional field production. Naik et al. (2019) used PCE for a polymer injection case. However, all published methods treat the time-dependent response in a discontinuous manner. In other words, every timestep (or time interval) has its separate low-fidelity model, which is a function of the spatial and time-independent parameters. The PCE method has been used to serve as the forward model when performing history matching in a water flooding reservoir using the MCMC approach (Santoso et al. 2021a, 2021b). This approach develops inaccurate time-response interpolation and inefficiencies (Chinesta et al. 2017; Santoso et al. 2019a, 2019b). Therefore, there is a need to develop a new method to efficiently construct continuous time response.

Most published studies use Gaussian distribution for the informative prior (Marzouk et al. 2007; Elsheikh et al. 2012; Hamdi et al. 2017). In geoscience, informative prior is hard to obtain as the measurement data are often sparse and limited. Naik et al. (2019) performed the Bayesian MCMC with symmetric noninformative prior. Symmetric noninformative prior can lead to wrong solutions if the actual solution is not around the mean. Therefore, there is a critical need to develop a Bayesian framework that can adapt to any prior condition.

The outline of this paper is the following: In the section "Proposed Workflow," we introduce the proposed workflow and fundamental theories. In the section "Problem Formulation," we describe the problem of water injection in an oil reservoir. The section "Simulation Model" shows the reservoir model, the physical phenomena, and prior distributions. In the section "Results," we present the history-matching results. The section "Sensitivity Analysis" shows the sensitivity analysis of the LSTM's architecture, hyperparameters, training set, number of chains, and chain lengths. This section also compares the performance of RF prediction using different surrogate methods with different prior information. A test case with perturbed responses and a test case with uncertainty parameters with narrowed range are presented. In the section "Discussions," we discuss the workflow's results. Finally, we deliver the conclusions of this study in the section "Conclusions."

---

## Proposed Workflow

### LSTM Network

LSTM is a low-fidelity model or surrogate that allows the construction of time-continuous responses. The LSTM is a variant of a recurrent neural network for capturing both short-term and long-term temporal functions (Hochreiter and Schmidhuber 1997; Jiang and Durlofsky 2020). It establishes the correlation between different quantities and different timesteps.

An LSTM network consists of repeating units, as shown in Fig. 1. The building unit of LSTM consists of four components: input gate **i**, forget gate **f**, cell candidate **g**, and output gate **o**. Each gate uses a sigmoid or tanh (hyperbolic tangent) activation function. It involves two-state parameters passed through multiple layers to establish temporal correlation: cell state **c** and hidden state **M**. This study considers sequential inputs of multiple variables and sequential outputs, following the approach by Hochreiter and Schmidhuber (1997), and a detailed derivation can be found in this paper.

For instance, in the test study discussed later, there are 11 input variables corresponding to oil viscosity, residual water saturation, residual oil saturation, relative permeability curves, permeability and porosity correlations, injection rate with n = θ₁, θ₂, …, θ₁₀₀ scenarios, timesteps, and sequential outputs corresponding to the RF. In this case, **M** consists of 100 timesteps. The dimension of the sequential inputs **θ** and sequential outputs **M** are, therefore, 11×100 and 1×100, respectively.

> **Fig. 1** — Typical LSTM network architecture where the red line denotes the flow of input parameters in the unit. The dark-blue lines represent the flow of hidden states (outputs) at every timestep, and the green lines indicate the flow of cell states at every timestep. *(figure not reproduced)*

In the implementation, the LSTM units are combined with fully connected layers, batch normalization layers, dropout layers, and regression or classification layers to form a standard LSTM network (Hochreiter and Schmidhuber 1997). It is further combined with convolutional layers from convolutional LSTM network to capture the spatial-temporal correlation (Shi et al. 2015). Zhang et al. (2022a) proposed a CNN-BiLSTM surrogate to help detect the CO₂ leakage. There could be other combinations with a U-Net to form an R-U-Net network for spatial-temporal problems (Tang et al. 2020). Since we are only dealing with temporal problems, we use the standard LSTM network.

In the implementation, the network's architecture should be optimized by adjusting the number of hidden LSTM units, fully connected layers, batch normalization layers, dropout layers, and the placement of the layers. The training process is conducted using the stochastic gradient descent method, in which the stochastic gradient is updated iteratively with minibatches. In our case, the minibatches are 2, and the total observations are 300, so we will update the stochastic gradient 150 times for one epoch, and 2 forward simulations are needed to calculate one stochastic gradient (Graves and Schmidhuber 2005; Hochreiter and Schmidhuber 1997). The hyperparameters, including the learning rate, number of epochs, and batch size should therefore be optimized. The number and quality of training sets also determine the success of the training (Bishop 2006; Santoso et al. 2019a, 2019b; Zhang et al. 2022b). The training sets should cover all possible ranges and variations without overfitting the network training (He et al. 2020; Santoso et al. 2020).

We use the mean squared error as the objective function when deploying LSTM, which is defined by:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}\left(RF_i - RF_{i,pred}\right)^2 \tag{1}
$$

where $n$ is the number of time intervals, $RF_i$ and $RF_{i,pred}$ are, respectively, the reference RF and the predicted RF at time interval $i$.

### Bayesian Optimization

LSTM network requires careful tuning of hyperparameters and architecture to converge in the training process and accurately establish temporal correlation. However, the tuning process is exhaustive and expensive if it relies on a trial-and-error process. In this work, we use Bayesian optimization to tune the LSTM network. The Bayesian optimization aims to maximize an objective function, defined as follows,

$$
\mathbf{x}^{*} = \arg\max_{\mathbf{x}\in\mathbb{R}^{N_{opt}}} f(\mathbf{x}) \tag{2}
$$

where $\mathbf{x}$ represents the parameters to be optimized, $N_{opt}$ is the number of optimized parameters, and $f(\mathbf{x})$ is the objective function. The formulation of Bayesian optimization is based on Bayes's rule in Eq. 5, in which the posterior distribution (optimized parameters) is obtained by updating the prior distribution. There are three critical elements within the workflow: the use of the acquisition function to update the priors, the use of Gaussian process (GP) to assist the calculation of the acquisition function, and the selection of the covariance function to calculate the GP.

GP is an extension of the multivariate Gaussian distribution to an infinite-dimensional stochastic process for which any finite combination of dimensions will be a Gaussian distribution (Brochu et al. 2010; Rasmussen and Williams 2006). GP provides a prediction in the form of distribution. The GP is defined as follows:

$$
g(\mathbf{x}) = GP\left(\mu(\mathbf{x}),\, K(\mathbf{x},\mathbf{x}')\right) \tag{3}
$$

where $g(\mathbf{x})$ returns a scalar for an arbitrary $\mathbf{x}$, $\mu(\mathbf{x})$ is the mean, and $K(\mathbf{x},\mathbf{x}')$ is covariance function.

The computation process can be conducted using sequential design for optimization algorithm (Brochu et al. 2010; Cox and John 1992) or Monte Carlo integration (Wilson et al. 2018). The Bayesian optimization has been successfully deployed in various applications, such as tuning the hyperparameters of ANN (Albattat et al. 2022; He et al. 2021c, 2021d, 2022c) and LSTM (He et al. 2021e; Santoso et al. 2021a, 2021b), which inspire us to deploy the Bayesian optimization in this study. He et al. (2022b) has used the Bayesian optimization to optimize different surrogates to investigate the relationship between pore scale parameters and field scale responses.

### Bayesian MCMC

Consider the forward problem represented by,

$$
\mathbf{Y} = \mathcal{M}(\boldsymbol{\theta}) + \varepsilon,\quad \varepsilon \in \mathbb{R}^{N_{out}} \tag{4}
$$

where $\boldsymbol{\theta}$ is a set of $N_{in}$ dimensional input parameters, $\mathbf{Y}$ is a set of $N_{out}$ dimensional observations or experimental data, $\mathcal{M}(\boldsymbol{\theta})$ is a forward model to predict $\mathbf{Y}$ as a function of $\boldsymbol{\theta}$, and $\varepsilon$ describes the $N_{out}$ dimensional errors between observations $\mathbf{Y}$ and predictions $\mathcal{M}(\boldsymbol{\theta})$. We use a Gaussian discrepancy with zero mean and diagonal covariance matrix of the form $\Sigma = \sigma^2 \mathbf{I}_{N_{out}}$ with residual variance $\sigma^2 = Var(\varepsilon_i),\ i = 1,\dots,N_{out}$, to represent $\varepsilon \sim \mathbb{N}(\varepsilon \mid 0, \Sigma)$. This discrepancy term represents the measurement error and model inaccuracy (Wagner et al. 2019).

The Bayesian inversion estimates the posterior distribution of model parameters $\boldsymbol{\theta}$ by the following equation,

$$
P(\boldsymbol{\theta}\mid\mathbf{Y}) = \frac{L(\boldsymbol{\theta};\mathbf{Y})\,P(\boldsymbol{\theta})}{\int_{\boldsymbol{\theta}} L(\boldsymbol{\theta};\mathbf{Y})\,P(\boldsymbol{\theta})\,d\boldsymbol{\theta}} \tag{5}
$$

where $P(\boldsymbol{\theta})$ is the prior distribution that describes the beliefs of the parameters before model inversion, $L(\boldsymbol{\theta};\mathbf{Y})$ is the likelihood function (Wagner et al. 2019), $\int L(\boldsymbol{\theta};\mathbf{Y})P(\boldsymbol{\theta})d\boldsymbol{\theta}$ is the evidence, and $P(\boldsymbol{\theta}\mid\mathbf{Y})$ is the posterior distribution that combines the beliefs and the observations $\mathbf{Y}$.

The Bayesian inversion is solved using MCMC. The fundamental idea is to build a Markov chain $(\boldsymbol{\theta}^1, \boldsymbol{\theta}^2, \dots)$ over the prior support $\mathcal{D}$ with an invariant distribution that equals the posterior distribution of the interest. The Markov chain is defined by a transition probability $K(\boldsymbol{\theta}^{t+1}\mid\boldsymbol{\theta}^{t})$ from $\boldsymbol{\theta}^t$ of a chain at iteration step $t$ to the $\boldsymbol{\theta}^{t+1}$ at iteration step $t+1$ (Wagner et al. 2019). The posterior (next iteration) distribution of a Markov chain is written as,

$$
P(\boldsymbol{\theta}^{t+1}\mid\mathbf{Y}) = \int P(\boldsymbol{\theta}^{t}\mid\mathbf{Y})\,K(\boldsymbol{\theta}^{t+1}\mid\boldsymbol{\theta}^{t})\,d\boldsymbol{\theta}^{t} \tag{6}
$$

The integral in Eq. 6 is calculated using Monte Carlo integration.

### AIES Algorithm

The four most used MCMC algorithms are Metropolis-Hastings, adaptive Metropolis, Hamiltonian Monte Carlo, and AIES algorithms. However, the first three algorithms require extensive tuning to improve the performance of the convergence process (Wagner et al. 2019). Goodman and Weare (2010) recommended AIES to overcome this issue. The advantage of AIES algorithm is that the target distribution remains invariant.

The stretch move is used to achieve affine invariance. The new candidate is generated by the following equation:

$$
\boldsymbol{\theta}_i' = \boldsymbol{\theta}_j^{t} + Z\left(\boldsymbol{\theta}_i^{t} - \boldsymbol{\theta}_j^{t}\right),\qquad Z \sim p(z) =
\begin{cases}
\dfrac{1}{\sqrt{z}}\,\dfrac{1}{2\left(\sqrt{a} - 1/\sqrt{a}\right)} & \text{if } z \in [1/a,\, a] \\[2mm]
0 & \text{otherwise}
\end{cases} \tag{7}
$$

We need to sample from the distribution $p(z)$ controlled by only one tuning parameter $a > 1$. This tuning parameter is usually set to 2 (Goodman and Weare 2010). We use the following process (Algorithm 1) to perform the iteration $\boldsymbol{\theta}^t \to \boldsymbol{\theta}^{t+1}$, using one stretch move per walker, where $L$ is the number of walkers.

**Algorithm 1 — AIES algorithm**

```
for k = 1, ..., L
    Choose j ∈ {1..L} \ {k} at random
    Generate θ' = θ_j^t + Z (θ_k^t − θ_j^t),  all Z choices independent
    Accept, set θ_k^{t+1} = θ', with probability
        min{ 1,  z^(N_in − 1) · P(θ' | Y) / P(θ_k^t | Y) }
    Otherwise reject, set θ_k^{t+1} = θ_k^t
end
```

### Proposed Workflow

The proposed history-matching workflow is illustrated in Fig. 2. The process consists first of identifying the uncertainty parameters and their prior distributions, followed by the construction of surrogate LSTM using a Latin hypercube DoEs (shaded box). This process includes LSTM training and verification. Bayesian MCMC loop is then applied using LSTM surrogate. The high-fidelity models (simulations) are then used to ensure accurate and physically consistent history matching. This step includes an inspection to repeat the LSTM construction process if the accuracy is below a required threshold. Observation mismatch is evaluated, and another inspection is applied if the history match is not satisfactory, which may require revisiting the uncertainty parameters and their ranges.

> **Fig. 2** — Illustration of the proposed workflow that combines Bayesian MCMC and LSTM network for assisted reservoir history matching. *(figure not reproduced)*

The four main steps of the workflow include (1) identification of uncertainty parameters and their prior distributions, (2) construction of surrogate LSTM using Bayesian optimization, (3) application of Bayesian MCMC, and (4) quality assessment. These four steps are detailed below.

**Step 1 – Identifying uncertainty parameters:** The uncertainty (matching) parameters are assessed based on the data and existing knowledge. Each parameter should be described with its range of variability (upper and lower bounds) and distribution. If there is not enough information, uniform distribution could be assigned. This step is important to provide initial prior distribution for the history-matching process. Revisiting this step could be needed to readjust the range of parameters if the model predictions do not enclose the observed data.

**Step 2 – Constructing LSTM surrogate model:** LSTM is used as a surrogate model to improve the efficiency and computational cost of Bayesian MCMC in performing forward predictions. LSTM is constructed using the initial prior distributions. The hyperparameters, architecture, and training-sample optimization is highly nonlinear (DeCastro-García et al. 2019). To overcome this challenge, the optimization process is performed in two steps. We use a two-loop approach to optimize both hyperparameters and training samples, as illustrated in Fig. 3. The first loop uses Bayesian optimization to optimize hyperparameters under constant training samples. The second loop is to refine the training-sample space using Latin hypercube and use the Bayesian-optimized network to optimize the training samples. The hyperparameters include learning rate, batch size, number of neurons in each layer, and number of epochs. The architecture consists of a number of hidden units and combinations with other layers. Refinement of training samples means increasing the resolution at response space. The predictability check is evaluated using blind points.

> **Fig. 3** — Workflow illustrating the construction of the LSTM surrogate model using Bayesian optimization. *(figure not reproduced)*

**Step 3 – Application of Bayesian MCMC:** The application of Bayesian MCMC follows Algorithm 1. The likelihood calculations are performed by running the LSTM network and evaluating the mismatch with the data. The initial priors are fed into the LSTM. A good quality matching is achieved when the mean solutions are close to the data. We use UQLab software for performing the Bayesian MCMC (Marelli and Sudret 2014).

**Step 4 – Quality assessment:** Quality assessment is required to ensure an acceptable surrogate model and good history matching. The process includes the evaluation of the discrepancy between the data and output produced from the posterior distribution. We calculate the Root Mean Square Error (RMSE) between the observation and the predictions. A threshold of RMSE = 0.003 is used to accept or reject the match. If the accuracy is poor, we need to reevaluate the LSTM network. If the quality is still poor after reevaluation, the workflow recommends revisiting the initial prior ranges.

---

## Problem Formulation

To demonstrate the proposed workflow, we study the problem of oil recovery for a 3D heterogeneous oil reservoir at undersaturated conditions using water flooding. Water injection into the oil reservoir is governed by the saturation equation and generalized Darcy law (Hoteit and Firoozabadi 2008a, 2008b). The mass conservation in 3D space is expressed as follows,

$$
\frac{\partial}{\partial t}\left(\phi S_\eta\right) + \nabla\cdot\left(u_\eta\right) = Q_\eta,\qquad \eta = \{o, w\} \tag{8}
$$

where $t$ is time, $\phi$ is the porosity, $S_\eta$ is the fluid phase saturation, $u_\eta$ is the phase superficial velocity, and $Q_\eta$ is the phase sink/source term. The subscription $\eta$ denotes the phase where $o$ is oil and $w$ is water. The number of mass conservation equations follows the number of phases, $N_{phase}$. The phase saturation is constrained as follows,

$$
\sum_{i(\eta)}^{N_{phase}} S_i = 1 \tag{9}
$$

The superficial velocity for each phase is modeled using extended Darcy's law, given by,

$$
u_\eta = -\frac{\mathbf{k}\,k_{r\eta}}{\mu_\eta}\nabla\Phi_\eta \tag{10}
$$

where $\mathbf{k}$ is the absolute permeability of the reservoir, $k_{r\eta}$ is the phase relative permeability, $\mu_\eta$ is the phase viscosity, and $\Phi_\eta$ is the phase potential, which is given by

$$
\Phi_\eta = P_\eta + \rho_\eta g z \tag{11}
$$

where $P_\eta$ is the phase pressure, $g$ is the gravitational acceleration, and $z$ is the phase hydrostatic height.

### Dimensionless Variables

For a reservoir undergoing water injection, the time-dependent oil RF relative to the initial oil-in-place is given by,

$$
RF(t) = \frac{\int_0^t Q_{po}(t)\,dt}{S_o \cdot PV / B_o} \tag{12}
$$

where $Q_{po}(t)$ is the production rate, $PV$ is reservoir pore volume (PV), $B_o$ is the oil FVF, and $S_o$ is the initial oil saturation in the reservoir. The unknowns are pressure and saturation. The two-phase fluid flow problem is solved implicitly using the CMG-IMEX simulator (CMG 2016).

To be physically scalable for the different static and dynamic conditions of a reservoir, we introduce a dimensionless time, defined by,

$$
t_D = \frac{\int_0^t Q_{iw}(t)\,dt}{PV / B_w} \tag{13}
$$

where $Q_{iw}$ is the water injection rate and $B_w$ is the water formation volume factor. The dimensionless time represents the cumulative amount of injected water into the reservoir relative to the reservoir PV.

---

## Simulation Model

To demonstrate the Bayesian-LSTM history-matching workflow, we developed a synthetic history-matching problem based on the 10th SPE comparative solution model, which represents an inverted five-spot pattern undergoing water injection (see Fig. 4). The details about fluid properties, rock properties, and rock-fluid properties can be found in Christie and Blunt (2001). We use the upper layers (Tarbert formation), representing a highly heterogeneous nearshore environment. The fine-scale model has 60×220×35 grid blocks, each of dimensions of 20×10×2 ft³ in *x*-, *y*-, and *z*-directions, respectively. The following equations are used to represent the relative permeability curves (Bradley 1987):

$$
K_{rw} = K_{rw\max}\,(S_e)^{w_{exp}},\qquad K_{ro} = K_{ro\max}\,(S_e)^{o_{exp}} \tag{14}
$$

$$
S_e = \frac{S_w - S_{rw}}{1 - S_{rw} - S_{ro}} \tag{15}
$$

> **Fig. 4** — 3D model of the reservoir showing an inverted five-spot pattern with permeability distribution of the base-case model in 3D (left) and 2D cross sections (right). *(figure not reproduced)*

We select 10 static and dynamic uncertainty parameters in the history-matching problem, which are used to generate the training samples for the 3D model. The uncertainty parameters are provided with their ranges in Table 1, which include the maximum water relative permeability endpoint $K_{rw\max}$, oil relative permeability endpoint $K_{ro\max}$, the exponents, $w_{exp}$ and $o_{exp}$, oil viscosity $\mu_o$, irreducible water saturation $S_{rw}$, residual oil saturation $S_{ro}$, and water injection rates $Q_{inj}$. Porosity and permeability are illustrated in Fig. 5, which is described as follows,

**Table 1** — The range of the parameters and the value of base-case parameters.

| Parameters | Base case | Range |
|---|---|---|
| Water relative permeability endpoint, $K_{rw\max}$ | 1 | 0.7–1 |
| Oil relative permeability endpoint, $K_{ro\max}$ | 1 | 0.7–1 |
| Exponents, $w_{exp}$ | 2 | 1–3 |
| Exponents, $o_{exp}$ | 2 | 1–3 |
| Oil viscosity, $\mu_o$ (cp) | 3 | 1–4 |
| Irreducible water saturation, $S_{rw}$ | 0.2 | 0.1–0.3 |
| Residual oil saturation, $S_{ro}$ | 0.2 | 0.1–0.3 |
| Water injection rate, $Q_{inj}$ (bbl/day) | 5,000 | 4,000–6,000 |
| Correlation coefficients, $A$ | 25 | 20–30 |
| Correlation coefficients, $B$ | −1.6 | −3 to −1 |

> **Fig. 5** — Crossplot showing the relationship between the permeability and porosity of the base-case model (trendline: LogK = 25.6φ − 1.6). *(figure not reproduced)*

$$
\log K = A\phi + B \tag{16}
$$

where $K$ is the permeability, $\phi$ is the porosity, $A$ and $B$ are the correlation coefficients. Here we assume the above 10 parameters remain constant during the recovery process, and the RF changes with time. Therefore, the time steps are classified as a dynamic parameter while other parameters are classified as static parameters. The ranges of variation of $A$ and $B$ in the history match problem are provided in Table 1. Before training the network and performing the history matching, we normalize the uncertainty parameters to (−1, 1) using a min-max normalization approach to enhance the accuracy.

The base-case model is set to inject water at a constant rate of 5,000 bbl/day and a maximum bottomhole pressure of 10,000 psi, as described in Christie and Blunt (2001). The bottomhole pressure at the producers is kept constant at 4,000 psi.

Fig. 6 shows water saturation distribution after the injection of different PVs of water (i.e., at dimensionless times, $t_D$ = 0.01, 0.1, 0.5, and 1.0) into the reservoir. Water channeling is observed, which confirms the significant effect of rock heterogeneity in this problem. Fig. 7 shows the oil RF (left) and oil production rate (right) vs. dimensionless time. The RF is about 30% after injection 1 PV of water (i.e., $t_D$ = 1.0). The profile of oil production rate indicates a peak at about $t_D$ = 0.1, which corresponds with the occurrence of significant water breakthrough. This simulation case represents a typical behavior of the model, which serves as the baseline case in this study. The parameters corresponding to the baseline case are shown in Table 1.

> **Fig. 6** — Water saturation at different dimensionless times, corresponding to $t_D$ = 0.01, 0.1, 0.5, and 1.0, that is, after 1, 10, 50, and 100% of water PV injections, which show a significant effect of heterogeneity and water channeling. *(figure not reproduced)*

> **Fig. 7** — Simulation results showing the RF (left) and oil production rate (right) vs. dimensionless time from the model baseline case. *(figure not reproduced)*

---

## Results

We created synthetic history match data to serve as the observed data, based on the matching case as provided in Table 2. No prior knowledge is assumed for the parameters used to generate the observed data. We add an unknown Gaussian discrepancy with zero mean value and covariance matrix $\Sigma$ to represent the observation error, and the error can be calculated during the inversion process. The objective here is to demonstrate the proposed workflow to history match the matching case.

**Table 2** — Comparison between the matching case and the mean values of the posterior.

| Parameters | Matching Case | Mean Value of the Posterior | Relative Error (%) |
|---|---|---|---|
| Water relative permeability endpoint, $K_{rw\max}$ | 0.8 | 0.77 | 3.8 |
| Oil relative permeability endpoint, $K_{ro\max}$ | 0.7 | 0.72 | 2.9 |
| Exponents, $w_{exp}$ | 2 | 1.72 | 14.0 |
| Exponents, $o_{exp}$ | 2.5 | 2.7 | 8.0 |
| Oil viscosity, $\mu_o$ (cp) | 2.2 | 2.4 | 9.1 |
| Irreducible water saturation, $S_{rw}$ | 0.25 | 0.25 | 0.0 |
| Residual oil saturation, $S_{ro}$ | 0.25 | 0.22 | 12.0 |
| Water injection rate, $Q_{inj}$ (bbl/day) | 4,500 | 4,135.9 | 8.1 |
| Correlation coefficients, $A$ | 27 | 26.24 | 2.8 |
| Correlation coefficients, $B$ | −2 | −1.43 | 28.5 |

The range of the uncertainty parameters are given in Table 1. The LSTM is then trained with 300 cases and tested with 100 randomly generated samples based on Latin hypercube DoE. We use Bayesian optimization to tune the LSTM instead of the traditional trial-and-error approach. The initial range of the hyperparameters and the optimized hyperparameters are shown in Table 3.

**Table 3** — Initial ranges of the hyperparameters and the optimized hyperparameters in the model.

| Hyperparameters | Initial Range | Optimized Hyperparameters |
|---|---|---|
| Neurons of the first LSTM layer | 200–800 | 588 |
| Neurons of the second LSTM layer | 200–800 | 578 |
| Neurons of the first fully connected layer | 50–200 | 91 |
| Epochs | 30–500 | 249 |
| Batch size | 1–300 | 2 |
| Learning rate | 0.00001–0.4 | 0.00144 |

We also performed sensitivity analysis on the used architecture, hyperparameters, and the number of training sets to obtain the optimum LSTM. The achieved optimum architecture of the LSTM and hyperparameter consists of 588 hidden units for the first LSTM layer, 578 hidden units for the second LSTM layer, 91 output weights for the first fully connected layer, 0.00144 learning rate, 249 epochs, and 2 batch sizes (summarized in Fig. 8). The optimum number of training sets is 300, where the dimension of the weights **w** and bias **b** are shown in Fig. 8. Additional details about the selection of this architecture are provided in the following sensitivity analysis section.

> **Fig. 8** — Optimum LSTM architecture for the test case under study. Input θ ∈ R^(11×100) → LSTM 1 (588 units) → LSTM 2 (578 units) → Fully-connected (91 weights) → Fully-connected → Output M(θ) ∈ R^(1×100). *(figure not reproduced)*

Fig. 9a shows the accuracy of LSTM in reproducing the RF profiles for cases from the training set, and Fig. 9b shows the accuracy of the LSTM in predicting cases that are not part of the training set. Figs. 9c and 9d show the parity plots evaluated at $t_D$ = 0.5 and $t_D$ = 1.0, respectively, where good quality training for the LSTM was achieved.

> **Fig. 9** — Comparison of RF vs. time generated by simulations and LSTM surrogate, where (a) shows cases selected from the training set, (b) compares blind cases that are not included in the training set, and (c) and (d) parity plots evaluated at $t_D$ = 0.5 and $t_D$ = 1.0, respectively. *(figure not reproduced)*

The MCMC is tuned with the Bayesian optimization algorithm to replace the trial-and-error process. We explore 1 to 200 chains and 100 to 10,000 lengths per chain for the Bayesian runs and burn in 50% of the results. The sensitivities are discussed in the following sensitivity analysis section. The optimum scenario is obtained with 15 chains and 1,200 lengths per chain, which accurately matches the mean prediction and the reference RF. The RF posterior prediction from the Bayesian runs is shown in Fig. 10a, while Fig. 10b shows the RMSE between the posterior predictions of RF from Bayesian MCMC runs and the reference (real) data. The distribution in blue color represents the uncertainties.

> **Fig. 10** — Comparison (a) and the RMSE (b) between the posterior prediction of RF from Bayesian MCMC runs and the reference data. *(figure not reproduced)*

Fig. 11 gives a 10×10 scatter plot matrix, where the diagonal elements visualize the posterior distribution of the input parameters, represented by histograms. The lower off-diagonal elements illustrate the pairwise scatter plots to show the correlation of the input parameters. Most of them are partially correlated to each other. The upper-off elements are empty. The prior and posterior distributions smoothed by the Kernel Density Estimator (KDE) of matching parameters are shown in Fig. 12. KDE can infer the univariate distribution using a nonparametric approach, and we use a standard Gaussian distribution as the kernel of the KDE. Here, we use KDE to generate the corresponding probability density function for each uncertainty parameter to help determine the most promising value for the inversed uncertainty parameters. The comparison between the matching case and the mean value of the posterior is shown in Table 2, where the obtained posteriors' values are very consistent with the reference data. Note that the parameters that show higher errors, such as the correlation coefficient $B$, are less sensitive to the objective function, and consequently, they are harder to match with better accuracy.

> **Fig. 11** — Scatter plot matrix for the uncertainty parameters: (a) prior samples and (b) posterior samples. The off-diagonal elements show the posterior distribution of each parameter, and the lower off-diagonal elements illustrate the pairwise scatter plots. *(figure not reproduced)*

We took the mean value from each posterior distribution and plugged them into the high-fidelity model. Fig. 13 shows the comparison with the high-fidelity model. The mean prediction from the Bayesian is in excellent agreement with the reference RF data, which demonstrates the robustness of the Bayesian approach.

During the Bayesian MCMC process, we needed to call and execute the forward model about 18,000 times in this case, which is impractical to be performed on the high-fidelity model (simulator) that needs nearly half an hour per run. The forward simulation of the LSTM took about 0.1 seconds. About 18,000 iterations of LSTM forward simulation were needed to reach convergence of the Bayesian MCMC approach, which took 37 minutes. The training of the LSTM required 249 epochs with 150 iterations for each epoch (i.e., a total of 37,350 iterations), which corresponded to a training time of 33 minutes. On the other hand, the generation of the 300 training samples needed 9,000 minutes (serial high-fidelity simulations). Therefore, the overall time is about 9,080 minutes with the LSTM as the forward model. In contrast, if we use the high-fidelity model as the forward model (without LSTM), one needs about 540,000 minutes to perform the 18,000 high-fidelity simulations. Therefore, the saving in computational time is more than 98% for this case.

> **Fig. 12** — Prior and posterior distributions for the eight uncertainty variables, showing the convergence of the mean values, including the situation when the prior distributions are skewed relative to the posteriors. The injection rate and *B* exhibited similar (not shown here). *(figure not reproduced)*

> **Fig. 13** — Comparison of high-fidelity prediction using Bayesian's posteriors and the RF corresponding to the matching case (a) and the corresponding error between the two RFs (b). *(figure not reproduced)*

---

## Sensitivity Analysis

### Sensitivities on Architecture, Hyperparameters, and Number of Training Sets for LSTM

We emphasize that the hyperparameters (learning rate, number of epochs, and batch size) should be optimized simultaneously for the LSTM network, while the number and length of chains should be optimized simultaneously for the Bayesian inference. Therefore, we deploy the Bayesian approach to perform the optimization rigorously. This section aims to understand each hyperparameter's significance toward the LSTM network's performance and the Bayesian inference.

**Architecture.** We consider four architectures for the network (Fig. 14), where Architecture 1 with one LSTM layer and one fully connected layer, Architecture 2 with one LSTM layer and two fully connected layers, Architecture 3 with two LSTM layers and one fully connected layer, and Architecture 4 with two LSTM layers and two fully connected layers (same as the one shown in Fig. 8). We use 300 training samples. The Bayesian optimization is deployed for tuning the hyperparameters. Within the Bayesian optimization, the architectural parameter is treated as categorical input. Fig. 15 shows the training and prediction error.

> **Fig. 14** — Testing of various network architectures in the optimization processes: (a) 1 LSTM and 1 fully-connected layer; (b) 1 LSTM and 2 fully-connected layers; (c) 2 LSTM and 1 fully-connected layer; and (d) 2 LSTM and 2 fully-connected layers. *(figure not reproduced)*

> **Fig. 15** — The effect of changing architecture on training RMSE (a) and prediction RMSE (b). The hyperparameters in each architecture are optimized using the Bayesian approach. *(figure not reproduced)*

Consider Architecture 1 to be a baseline case. Adding a fully connected layer or an LSTM unit improves the early time ($t_D$ = 0–0.2) accuracy but reduces the late time ($t_D$ = 0.2–1.0) accuracy. Adding both fully connected layer and LSTM unit provides accurate correlation until late time and produces low RMSE.

**Learning Rate.** We vary the learning rate from 0.00001 to 0.01. We use the obtained optimized architecture (see Fig. 14d) with 300 training sets, 249 epochs, and 2 batch sizes. Appendix A provide more details. Training RMSE and prediction RMSE decrease by increasing the learning rate at values below 0.00144, and that is because the LSTM is easy to trap at local minima when the learning rate is too low. Above 0.00144 learning rate, the RMSE starts to increase because a larger learning rate helped the LSTM to skip the global minima. Therefore, the obtained optimum learning rate for the network is 0.00144.

**Number of Epochs.** We vary the number of epochs from 50 to 1,000. We use the optimized architecture (Fig. 14d) with 300 training sets, 0.00144 learning rate, and 2 batch sizes. Appendix A provides more details. Increasing the number of epochs does not necessarily decrease the training and prediction RMSE. Small epochs are not enough to train the LSTM while large epochs could cause overfitting. As there is no significant difference in RMSE among 249, 500, and 1,000 epochs and we have the minimum error at 249 batch size for the training process, we pick 249 epochs as the optimum.

**Number of Batch Size.** We vary the batch size from 1 to 100. We use the optimized architecture (Fig. 14d) with 300 training sets, 0.00144 learning rate, and 249 epochs. Appendix A provides more details. Above 2 batch size, increasing batch size increases both training and prediction RMSE. Below 2 batch size, decreasing batch size also increases training and prediction RMSE. The overall RMSE depicts 2 batch sizes as the optimum. Moreover, both the training and prediction are in poor condition during the early stage. However, the 5-batch size performs slightly better than 2 in the later stage.

**Training Set.** We vary the samples from 100 to 500. Following the workflow in Fig. 3, we use the architecture in the section "Results" and deploy Bayesian optimization to tune the hyperparameters for each training set. The training set is space filling generated using Latin hypercube. We found no significant improvement in RMSE between 300 and 500 training sets, and therefore we picked 300 training sets as the optimum. In Appendix A, we discuss the detailed process of selecting the optimum number of training sets.

### Sensitivities on Number of Chains and Chain Lengths in Bayesian MCMC

**The Number of Chains.** In the Bayesian MCMC runs, we used five chains and varied their length from 100 to 10,000. We use the Bayesian-optimized LSTM (Fig. 14d) as the low-fidelity model. Fig. 16 shows the sensitivities of the chain lengths toward RF uncertainties. The distributions in the dark blue represent the uncertainties.

> **Fig. 16** — The effect of chain lengths on RF prediction uncertainty: 100 (a), 1,200 (b), 2,000 (c), and 10,000 (d). *(figure not reproduced)*

Increasing the chain's length reduces the variance of uncertainties of RF prediction, but no significant improvement was observed after the chain length reached 1,200. Fig. 17 provides four trace plots and the corresponding four plots of the average values of four unknown parameters, which highlight the convergence behavior of the MCMC simulation, where $\pi$ represents the KDE. The unknowns converged when the chain lengths reached ~1,200, which is selected as the optimum. Note that increasing the chain's length does not always guarantee a good match between the mean prediction and the data.

> **Fig. 17** — Trace plots of the four unknown parameters when the chain's length achieves 1,200: (a) $K_{ro\max}$, (b) $B$, (c) $o_{exp}$, (d) $S_{rw}$, and corresponding plots of the average value of the four unknowns: (e) $K_{ro\max}$, (f) $B$, (g) $o_{exp}$, (h) $S_{rw}$, which highlight the convergence behavior. *(figure not reproduced)*

**Multiple Chains.** We set the chain's length to be 1,200 and varied the number of chains from 5 to 200. Figs. 18a through 18c show the RF with the corresponding uncertainties. Increasing the number of chains can slightly improve the matching between the mean prediction and the reference case. Fig. 18d gives the RMSE errors between the mean prediction and the reference RF, where there is no apparent difference between 15 and 200 chains. Therefore, we set 15 as the optimum number of chains.

> **Fig. 18** — Effect of the number of chains on RF prediction uncertainty using (a) 5 chains, (b) 50 chains, (c) 100 chains, and (d) the corresponding errors, including other cases with a different number of chains. *(figure not reproduced)*

### Comparison of Different Proxy Methods

We compare the LSTM surrogate model with other commonly used discrete proxy methods such as PCEs, kriging, and SVR. We set the number of training samples to 300. Figs. 19a through 19c show the obtained matching of PCE, kriging, and SVR to the RF. We note that the prediction of kriging in the training set achieves an excellent match. However, it gave a poor performance in prediction. Figs. 19e and 19f show the average training RMSE and prediction RMSE of different methods, respectively. We observe that the LSTM surrogate achieves the best predictability.

More analyses about the effect of narrowing the range of the uncertainties and different prior information are provided in Appendixes B and C, respectively.

> **Fig. 19** — The performance of the surrogate models in predicting the RF and compared to the reference solutions (a) PCE, (b) kriging, (c) SVR, and (d) the corresponding errors showing the best accuracy with LSTM. *(figure not reproduced)*

### Test Case with Perturbation Responses

To test the applicability of the proposed method for cases with a nonsmooth response (objective function), we applied the method for a case with a perturbed oil recovery profile, obtained by varying the bottomhole pressure of the injectors. In this test, we switched the bottomhole pressure between 2,600 and 9,000 psi every 0.1 PV of water injection. The LSTM is then trained with 240, validated with 30, and tested with 30 randomly generated samples based on Latin hypercube DoE. Fig. 20a shows various samples of prediction cases where LSTM provided excellent reproduction of the high-fidelity model. The history-matched RF posterior prediction from the Bayesian runs is shown in Fig. 20b, which also shows excellent agreement with the reference profile. Detailed information about this case is provided in Appendix D.

> **Fig. 20** — Applicability of the proposed method for cases with nonsmooth RF profiles using LSTM surrogate, where (a) shows blind cases from LSTM predictions that are not included in the training set and (b) compares the posterior prediction of RF from Bayesian MCMC runs and the reference data. *(figure not reproduced)*

---

## Discussions

- The accuracy of the LSTM model in representing the high-fidelity model is critical. This accuracy can be observed from the consistency of mean Bayesian response and high-fidelity response (using Bayesian results as inputs) in capturing the data. The LSTM model could produce a similar mean Bayesian response to the high-fidelity response. There are several observations that could be highlighted: Stacking multiple LSTM layers establishes a long time response correlation while adding a fully connected layer provides more weights for accurate regression (Pascanu et al. 2013; Shabbeer et al. 2019). Single LSTM layer cannot capture complex variation in time response compared to stack LSTM layers (Pascanu et al. 2013). On the other hand, having multiple stacks of LSTM layers without fully connected layers reduces the accuracy as the number of weights is insufficient to describe the variation of the model (Shabbeer et al. 2019).
- Learning rate governs the transfer of information to the network (Allen-Zhu et al. 2018; Hanin and Rolnick 2018). Decreasing the learning rate does not always increase the prediction accuracy. There is an optimum learning rate to obtain accurate predictions. However, learning-rate optimization should be conducted simultaneously with the other hyperparameters.
- The number of epochs controls the frequency of the whole training set being fed into the network. Multiple feedings of the training set into the network during the training are essential to obtain the proper weights. It is also important to capture the essential features and identify noises. An increasing number of epochs generally decreases the RMSE (Afaq and Rao 2020). However, using a large number of epochs could lead to overfitting or expensive training. The optimum number of epochs is when there is no significant reduction in RMSE.
- Batch size denotes part of the training set used to calculate the gradient. Batch size is strongly correlated with learning rate; therefore, both control the diffusion of information to the network (Kandel and Castelli 2020). An optimum batch size at a fixed learning rate gives the minor training and prediction RMSE.
- The optimization of hyperparameters (tuning) should be performed simultaneously. Bayesian optimization provides a rigorous framework to conduct the tuning. The critical part of this tuning process is to select the proper objective function. We suggest using the predictability of the network as the objective function to avoid overfitting.
- An increasing number of training sets aim to capture more variations since space filling is generated. Therefore, increasing the number of training sets in a space filling manner decreases the RMSE. The optimum number of the training set is when there is no significant reduction in RMSE.
- Proper selection of the number of chains and chain lengths determines the convergence and accuracy of the Bayesian MCMC results. Results showed that there is an optimum number of chains and chains' length to achieve an accurate mean prediction. The proper selection of these parameters ensures ergodicity (Andrieu and Robert 2001; Mahendran et al. 2011; Moore 2015; Saksman and Vihola 2010). Increasing the chain's length reduces the variance of the prediction and increases the precision (Link and Eaton 2012). However, using a long chain consumes more computational resources. The chain's length can be thinned when there is no significant change in prediction (Link and Eaton 2012).
- Using multiple chains provides better initial seeds; therefore, it may improve convergence. However, not all initial seeds may lead to precise converged values. It may be trapped in local optima in the multimodal posterior (Andrieu and Robert 2001). Therefore, an optimum number of chains are needed to achieve precise converged values.
- Both the numbers of chains and chain lengths affect the precision of the Bayesian MCMC result. The optimization should be conducted simultaneously on both parameters. One cannot rely only on multiple chains to explore the initial seeds and use short chain lengths as it will give poor convergence and vice versa (Mahendran et al. 2011). Therefore, there is an optimum combination of chains and chain lengths to obtain precise results.
- The posterior prediction through the high-fidelity model is to ensure high-quality history matching. An accurate posterior prediction through the high-fidelity model is mainly influenced by the predictive quality of the LSTM network and the accuracy of mean prediction in Bayesian runs.
- It takes half an hour for four CPUs to run a high-fidelity model in parallel and 37 minutes for one CPU to perform the Bayesian MCMC. A P2000 GPU is used for training the LSTM surrogate, which takes 33 minutes to complete the training of 249 epochs. Although it saves more than 98% computational time than the traditional approach, the high-fidelity simulations still consume about 9,000 minutes in our specific case, and the computational time may be higher when dealing with more complex responses as more training samples will be needed to obtain a well-trained LSTM. More details about the workflow are provided in Appendix E.

---

## Conclusions

We present a novel workflow to perform history matching, which combines Bayesian MCMC and LSTM. The LSTM network is used to represent the high-fidelity model, including continuous time response to reduce computational cost. The network is tuned using the Bayesian approach. We use multiple chains of Bayesian MCMC to provide better initial seeds. The posterior prediction through the high-fidelity model ensures the robustness and physical consistency of the results. The LSTM network accurately represents the high-fidelity model, provides a time-continuous response, and reduces the computational cost for Bayesian MCMC runs. The predictive quality of the network is influenced by the proper selection of architecture, hyperparameters, and training sets. Bayesian optimization provides a robust framework for tuning the LSTM network. The precision of Bayesian MCMC is determined by an optimum selection of the number of chains and chain length. The number of chains controls the initial seeds, while the chain's length influences the variance. The high-fidelity model ensures robustness and physical consistency of the history-matching results. The posterior prediction through the high-fidelity model is mainly influenced by the predictive quality of the LSTM network and the mean prediction of Bayesian results. The correlation among matching parameters reflects the physical processes in the model. The proposed LSTM achieved better performance than other traditional discrete proxy methods, such as PCE, kriging, and SVR, which demonstrates the potential of using LSTM as a robust proxy for time-continuous problems.

---

## Acknowledgment

We acknowledge CMG Ltd. for providing the IMEX academic license, KAUST for the support, and UQLab for the software license.

---

## Appendix A — Selection of the DoEs

We use different experimental design methods, including Halton design, Sobol design, and Latin hypercube sampling method. We use the obtained optimized architecture (see Fig. 14d) 249 epochs, 2 batch sizes, 0.00144 learning rate, and 100 training samples generated from different experimental design methods. Figs. A-1a through A-1c shows the training and prediction RMSE after changing the learning rate. Figs. A-1d through A-1f shows the training and prediction RMSE after changing the number of epochs. Figs. A-1g through A-1i shows the training and prediction RMSE after changing the batch size. Figs. A-2a through A-2c shows the RMSE of the training and testing samples under different training samples. Figs. A-2d through A-2f shows the RMSE of the training and testing samples under different experimental design methods. We can clearly see that the Latin hypercube sampling method achieves the lowest training errors, and the Latin hypercube sampling method and Sobol design achieve close prediction errors. Therefore, we choose Latin hypercube sampling method in this study.

> **Fig. A-1** — The sensitivities of training, prediction, and optimum conditions at $t_D$ = 0.5 resulting from varying the learning rate (a–c), number of epochs (d–f), and batch size (g–i). *(figure not reproduced)*

> **Fig. A-2** — The sensitivities of training, prediction, and optimum conditions at $t_D$ = 0.5 resulting from varying the number of training set (a–c) and different experimental design methods (d–f). *(figure not reproduced)*

---

## Appendix B — Test Case of Uncertainty Parameters with Narrowed Range

We retrain the model with narrowed ranges based on the results shown in Fig. 12. Table B-1 shows the original and narrowed range of the uncertainty parameters. However, we did not obtain better performance in the accuracy of the inversed uncertainty parameters, as shown in Table B-2. The mean errors of the uncertainty parameters changed from 8.92 to 9.095%. This can be explained in Fig. 17. We can see that the uncertainty parameters converge to a specific value once we sample enough steps. The results will not significantly improve even if we narrow down the uncertainty parameters range because the Markov chain reaches a steady state when we sample sufficient chain lengths.

**Table B-1** — The original and narrowed range of the uncertainty parameters.

| Parameters | Original Range | Narrowed Range |
|---|---|---|
| Water relative permeability endpoint, $K_{rw\max}$ | 0.7–1 | 0.7–0.83 |
| Oil relative permeability endpoint, $K_{ro\max}$ | 0.7–1 | 0.7–0.82 |
| Exponents, $w_{exp}$ | 1–3 | 1.5–3 |
| Exponents, $o_{exp}$ | 1–3 | 2–3 |
| Oil viscosity, $\mu_o$ (cp) | 1–4 | 2–3 |
| Irreducible water saturation, $S_{rw}$ | 0.1–0.3 | 0.23–0.3 |
| Residual oil saturation, $S_{ro}$ | 0.1–0.3 | 0.18–0.3 |
| Water injection rate, $Q_{inj}$ (bbl/day) | 4,000–6,000 | 4,000–5,000 |
| Correlation coefficients, $A$ | 20–30 | 23–28 |
| Correlation coefficients, $B$ | −3 to −1 | −2.2 to −1 |

**Table B-2** — Comparison among the results of the matching case, the mean values of the posterior without narrowed range, and the mean values of the posterior with narrowed range.

| Parameters | Matching Case | Mean Value of Posterior without Narrowed Range | Mean Value of Posterior with Narrowed Range | Relative Error without Narrowed Range (%) | Relative Error with Narrowed Range (%) |
|---|---|---|---|---|---|
| Water, $K_{rw\max}$ | 0.8 | 0.77 | 0.77 | 3.8 | 3.8 |
| Oil, $K_{ro\max}$ | 0.7 | 0.72 | 0.73 | 2.9 | 4.28 |
| Exponents, $w_{exp}$ | 2 | 1.72 | 1.7 | 14.0 | 15 |
| Exponents, $o_{exp}$ | 2.5 | 2.7 | 2.8 | 8.0 | 12.0 |
| Oil viscosity, $\mu_o$ (cp) | 2.2 | 2.4 | 2.3 | 9.1 | 4.5 |
| Irreducible water sat., $S_{rw}$ | 0.25 | 0.25 | 0.24 | 0.0 | 4.0 |
| Residual oil sat., $S_{ro}$ | 0.25 | 0.22 | 0.22 | 12.0 | 12.0 |
| Inject. rate, $Q_{inj}$ (bbl/day) | 4,500 | 4,135.9 | 4,800 | 8.1 | 6.67 |
| Coefficients, $A$ | 27 | 26.24 | 26 | 2.8 | 3.7 |
| Coefficients, $B$ | −2 | −1.43 | −1.5 | 28.5 | 25.0 |

---

## Appendix C — Comparison of Different Prior Information

We use the optimized LSTM network (249 epochs, 2 batch sizes, and 0.00144 learning rate) and optimized MCMC (15 chains and 1,200 lengths) to investigate the effect of different prior information. Specifically, we compared the noninformative prior (all uniform distribution) with informative prior (all triangular distribution) and hybrid informative prior (half uniform distribution and half triangular distribution). Table C-1 shows the comparison between the matching case and the mean values of the posterior under informative prior. Table C-2 gives the comparison between the matching case and the mean values of the posterior under hybrid informative prior. The mean errors (total errors divided by the number of parameters) of all the uncertainty parameters are 8.92% for noninformative prior, 7.63% for informative prior, and 8.025% for hybrid informative prior. These results indicate that the more prior information we have, the more accurate inversion we will obtain through the Bayesian inversion process. The probability density function for the triangular distribution is shown below:

$$
f(t) =
\begin{cases}
\dfrac{2\,(t-a)}{(b-a)(c-a)}, & a \le t \le c \\[2mm]
\dfrac{2\,(b-t)}{(b-a)(b-c)}, & c \le t \le b
\end{cases} \tag{C-1}
$$

where $t \in [a, b]$, $a$ is the lower bound, $b$ is the upper bound, and $c$ is the top of the triangle.

**Table C-1** — Comparison between the matching case and the mean values of the posterior under informative prior.

| Parameters | Matching Case | Prior Distribution (Triangular) [a b c] | Mean Value of the Posterior | Relative Error (%) |
|---|---|---|---|---|
| Water, $K_{rw\max}$ | 0.8 | [1 4 2.2] | 0.8 | 0 |
| Oil, $K_{ro\max}$ | 0.7 | [0.7 1 0.8] | 0.81 | 15.7 |
| Exponents, $w_{exp}$ | 2 | [1 3 2] | 1.8 | 10.0 |
| Exponents, $o_{exp}$ | 2.5 | [1 3 2.5] | 2.7 | 8.0 |
| Oil viscosity, $\mu_o$ (cp) | 2.2 | [1 4 2.2] | 2.1 | 4.5 |
| Irreducible water sat., $S_{rw}$ | 0.25 | [0.1 0.3 0.25] | 0.22 | 12.0 |
| Residual oil sat., $S_{ro}$ | 0.25 | [0.1 0.3 0.25] | 0.25 | 0 |
| Inject. rate, $Q_{inj}$ (bbl/day) | 4,500 | [4,000 6,000 4,500] | 5,000 | 11.1 |
| Correlation coefficients, $A$ | 27 | [20 30 27] | 27 | 0 |
| Correlation coefficients, $B$ | −2 | [−3 −1 −2] | −1.7 | 15 |

**Table C-2** — Comparison between the matching case and the mean values of the posterior under hybrid informative prior.

| Parameters | Matching Case | Prior Distribution | Mean Value of Posterior | Relative Error (%) |
|---|---|---|---|---|
| Water, $K_{rw\max}$ | 0.8 | Triangular [1 4 2.2] | 0.77 | 3.75 |
| Oil, $K_{ro\max}$ | 0.7 | Triangular [0.7 1 0.8] | 0.84 | 20 |
| Exponents, $w_{exp}$ | 2 | Uniform [1 3] | 1.8 | 10.0 |
| Exponents, $o_{exp}$ | 2.5 | Uniform [1 3] | 2.7 | 8.0 |
| Oil viscosity, $\mu_o$ (cp) | 2.2 | Triangular [1 4 2.2] | 2.3 | 4.5 |
| Irreducible water sat., $S_{rw}$ | 0.25 | Triangular [0.1 0.3 0.25] | 0.22 | 12.0 |
| Residual oil saturation, $S_{ro}$ | 0.25 | Triangular [0.1 0.3 0.25] | 0.25 | 0 |
| Inject. rate, $Q_{inj}$ (bbl/day) | 4,500 | Uniform [4,000 4,500] | 5,100 | 13.3 |
| Correlation coefficients, $A$ | 27 | Uniform [20 30] | 28 | 3.7 |
| Correlation coefficients, $B$ | −2 | Uniform [−3 −1] | −1.9 | 5 |

---

## Appendix D — Test Case with Perturbation Responses

In this appendix, we provide more information about the test case with perturbed recovery response. Fig. D-1a shows samples of training cases, and Fig. D-1b shows samples of prediction cases where LSTM provided excellent reproduction and predictions. Fig. D-1c shows the parity plots evaluated at $t_D$ = 0.5. Fig. D-1d indicates the low RMSE values for both the training and testing samples.

> **Fig. D-1** — Applicability of the proposed method for cases with nonsmooth RF profiles using LSTM surrogate, where (a) shows cases selected from the training set, (b) compares blind cases that are not included in the training set, (c) parity plots evaluated at $t_D$ = 0.5, and (d) RMSE for the training and testing cases. *(figure not reproduced)*

**Table D-1** — Initial ranges of the hyperparameters and the optimized hyperparameters in the model.

| Hyperparameters | Initial Range | Optimized Hyperparameters |
|---|---|---|
| Neurons of the first LSTM layer | 200–800 | 546 |
| Neurons of the second LSTM layer | 200–800 | 562 |
| Neurons of the first fully connected layer | 50–200 | 93 |
| Epochs | 30–500 | 295 |
| Batch size | 1–300 | 5 |
| Learning rate | 0.00001–0.4 | 0.0012312 |

Table D-1 shows the initial and optimized hyperparameters for the LSTM surrogate. We then performed the Bayesian MCMC to inverse the uncertainty parameters, and the uncertainty parameters are the same as in the previous matching case. The only difference is that we use a varying bottomhole pressure for the producers to generate a RF with perturbation. We use 15 chains and 1,200 lengths per chain to conduct the Bayesian MCMC process. The RF posterior prediction from the Bayesian runs is shown in Fig. D-2a, and Fig. D-2b shows the RMSE between the posterior prediction of RF from Bayesian MCMC runs and the real data. The distribution in blue color represents the uncertainties.

> **Fig. D-2** — Comparison (a) and the RMSE (b) between the posterior prediction of RF from Bayesian MCMC runs and the reference (real) data. *(figure not reproduced)*

The comparison between the matching case and the mean value of the posteriors is shown in Table D-2, where the obtained posteriors' values are very consistent with the reference data. The mean error (total errors divided by the number of parameters) in this case is 11.02%, which is slightly larger than the mean error of the uncertainty parameters that are inversed by smooth RF curve of 8.92%. The reason is that we use the same number of training samples to train the LSTM. However, a RF curve with perturbation is harder to train than a smooth RF curve. Therefore, some accuracy is lost during the training of the LSTM, leading to slightly larger errors.

**Table D-2** — Comparison between the matching case and the mean values of the posterior.

| Parameters | Matching Case | Mean Value of the Posterior | Relative Error (%) |
|---|---|---|---|
| Water relative permeability endpoint, $K_{rw\max}$ | 0.8 | 0.72 | 10.0 |
| Oil relative permeability endpoint, $K_{ro\max}$ | 0.7 | 0.72 | 11.4 |
| Exponents, $w_{exp}$ | 2 | 1.7 | 15.0 |
| Exponents, $o_{exp}$ | 2.5 | 2.4 | 4.0 |
| Oil viscosity, $\mu_o$ (cp) | 2.2 | 2.6 | 18.1 |
| Irreducible water saturation, $S_{rw}$ | 0.25 | 0.22 | 12.0 |
| Residual oil saturation, $S_{ro}$ | 0.25 | 0.26 | 4.0 |
| Water injection rate, $Q_{inj}$ (bbl/day) | 4,500 | 5,100 | 13.3 |
| Correlation coefficients, $A$ | 27 | 29 | 7.4 |
| Correlation coefficients, $B$ | −2 | −2.3 | 15.0 |

---

## Appendix E — Code Workflow for Bayesian Inversion

The Bayesian inversion mainly includes five steps as shown in Fig. E-1. Step 1: we first define the prior information of the uncertainty parameters including distribution type and ranges. Step 2: we then define the forward model to compute the likelihood during the posterior computation, here the forward model is the well-trained LSTM surrogate. Step 3: we need to define the parameters for the MCMC method such as the number and the length of chains, and we also need to specify the response that we want to match. Here the response is the RF of the matching case. Step 4: we then define the solver parameters and specify the MCMC algorithm, and we used the AIES algorithm in this work. Step 5: we finally run the Bayesian inversion process and post-process the posterior results. Specifically, we burn in 50% of the results to get a stable posterior distribution. The detailed code implementation can be found in UQLab.

**Fig. E-1** — The code workflow for the Bayesian inversion process.

```matlab
% Step 1 — Input prior information
% viscosity
PriorOpts.Marginals(1).Name = 'uo';
PriorOpts.Marginals(1).Type = 'Uniform';
PriorOpts.Marginals(1).Parameters = [1 4];

% Step 2 — Define forward model (LSTM surrogate)
ModelOpts.mFile = 'LSTM_Surrogate';
ModelOpts.isVectorized = false;
myForwardModel = uq_createModel(ModelOpts);

% Step 3 — Define parameters for MCMC
NChains = 15;
Steps = 1200;
% Define the response we want to match
myData.Name = 'Recovery_Factor';

% Step 4 — Define the solver
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES';
Solver.MCMC.NChains = NChains;
Solver.MCMC.Steps = Steps;

% Step 5 — Run the simulation
myBayesianAnalysis = uq_createAnalysis;
% Burn in
uq_postProcessInversion('burnIn', 0.5);
```

---

## References

- Afaq, S. and Rao, S. 2020. Significance Of Epochs On Training A Neural Network. *Int J Sci Technol Res* 19 (6): 485–488.
- Agarwal, B. and Blunt, M. J. 2003. Streamline-Based Method With Full-Physics Forward Simulation for History-Matching Performance Data of a North Sea Field. *SPE J.* 8 (2): 171–180. SPE-84952-PA. https://doi.org/10.2118/84952-PA.
- Agarwal, B., Hermansen, H., Sylte, J. E. et al. 2000. Reservoir Characterization of Ekofisk Field: A Giant, Fractured Chalk Reservoir in the Norwegian North Sea-History Match. *SPE Res Eval & Eng* 3 (6): 534–543. SPE-68096-PA. https://doi.org/10.2118/68096-PA.
- AlAmeri, S., AlBreiki, M., and Geiger, S. 2020. History Matching Under Geological Constraints Coupled with Multiobjective Optimization To Optimize MWAG Performance: A Case Study in A Giant Onshore Carbonate Reservoir in the Middle East. *SPE Res Eval & Eng* 23 (2): 534–550. SPE-196715-PA. https://doi.org/10.2118/196715-PA.
- Albattat, R., He, X., AlSinan, M. et al. 2022. Modeling Lost-Circulation in Fractured Media Using Physics-Based Machine Learning. Paper presented at the 83rd EAGE Annual Conference & Exhibition, Madrid, Spain, 5–8 June. https://doi.org/10.3997/2214-4609.202210204.
- Allen-Zhu, Z., Li, Y., and Song, Z. 2018. A Convergence Theory for Deep Learning via Over-Parameterization. Paper presented at the International Conference on Machine Learning, Stockholm, Sweden, 10–15 July.
- Andrieu, C. and Robert, C. P. 2001. Controlled MCMC for Optimal Sampling. Working Papers. 2001-33.
- Awemo, K. N., Ajala, I., Schwietzer, C. A. et al. 2014. P05 Assisted History Matching with Application of Adjoint Method Sensitivity Computation: Case Study North German Basin Oilfield. Paper presented at the 2nd EAGE Integrated Reservoir Modelling Conference, Dubai, UAE, 16–19 November.
- Bishop, C. 2006. *Pattern Recognition and Machine Learning*, first edition. New York, USA: Springer-Verlag.
- Bradley, H. B. 1987. *Petroleum Engineering Handbook*. Richardson, Texas, USA: SPE.
- Brochu, E., Cora, V. M., and De Freitas, N. 2010. A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. ArXiv:1012.2599.
- Caers, J. 2003. Efficient Gradual Deformation Using a Streamline-Based Proxy Method. *J Pet Sci Eng* 39 (1–2): 57–83. https://doi.org/10.1016/S0920-4105(03)00040-8.
- Caers, J. and Hoffman, T. 2006. The Probability Perturbation Method: A New Look at Bayesian Inverse Modeling. *Math Geol* 38 (1): 81–100. https://doi.org/10.1007/s11004-005-9005-9.
- Chavent, G., Dupuy, M., and Lemmonier, P. 1975. History Matching by Use of Optimal Theory. *SPE J.* 15 (1): 74–86. SPE-4627-PA. https://doi.org/10.2118/4627-PA.
- Chen, W. H., Gavalas, G. R., Seinfeld, J. H. et al. 1974. A New Algorithm for Automatic History Matching. *SPE J.* 14 (6): 593–608. SPE-4545-PA. https://doi.org/10.2118/4545-PA.
- Chinesta, F., Huerta, A., Rozza, G. et al. 2017. Model Reduction Methods. In *Encyclopedia of Computational Mechanics*, second edition, 1–36. Chichester, UK: John Wiley & Sons, Ltd. https://doi.org/10.1002/9781119176817.ecm2110.
- Christie, M. A. and Blunt, M. J. 2001. Tenth SPE Comparative Solution Project: A Comparison of Upscaling Techniques. *SPE Res Eval & Eng* 4 (4): 308–317. SPE-72469-PA. https://doi.org/10.2118/72469-PA.
- CMG. 2016. *IMEX User Guide*. Calgary, Canada: Computer Modelling Group Ltd.
- Cox, D. D. and John, S. 1992. A Statistical Method for Global Optimization. In IEEE International Conference on Systems, Man, and Cybernetics, 1241–1246. Piscataway, New Jersey, USA: Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/ICSMC.1992.271617.
- DeCastro-García, N., Muñoz Castañeda, Á. L., Escudero García, D. et al. 2019. Effect of the Sampling of a Dataset in the Hyperparameter Optimization Phase over the Efficiency of a Machine Learning Algorithm. *Complexity* 2019: 1–16. https://doi.org/10.1155/2019/6278908.
- Elsheikh, A. H., Jackson, M. D., and Laforce, T. C. 2012. Bayesian Reservoir History Matching Considering Model and Parameter Uncertainties. *Math Geosci* 44 (5): 515–543. https://doi.org/10.1007/s11004-012-9397-2.
- Feraille, M., Roggero, F., Manceau, E. et al. 2003. Application of Advanced History Matching Techniques to an Integrated Field Case Study. Paper presented at the SPE Annual Technical Conference and Exhibition, Denver, Colorado, USA, 5–8 October. SPE-84463-MS. https://doi.org/10.2118/84463-MS.
- Goodman, J. and Weare, J. 2010. Ensemble Samplers with Affine Invariance. *Comm App Math Comp Sci* 5 (1): 65–80. https://doi.org/10.2140/camcos.2010.5.65.
- Graves, A. and Schmidhuber, J. 2005. Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures. *Neural Netw* 18 (5–6): 602–610. https://doi.org/10.1016/j.neunet.2005.06.042.
- Hamdi, H., Sousa, M. C., and Behmanesh, H. 2017. Bayesian History-Matching and Probabilistic Forecasting for Tight and Shale Wells. Paper presented at the SPE Unconventional Resources Conference, Calgary, Alberta, Canada, 15–16 February. SPE-185082-MS. https://doi.org/10.2118/185082-MS.
- Hanin, B. and Rolnick, D. 2018. How to Start Training: The Effect of Initialization and Architecture. In *Advances in Neural Information Processing Systems*, 571–581. California, USA: Neural Information Processing Systems Foundation.
- Haugen, V., Nævdal, G., Natvik, L.-J. et al. 2008. History Matching Using the Ensemble Kalman Filter on a North Sea Field Case. *SPE J.* 13 (4): 382–391. SPE-102430-PA. https://doi.org/10.2118/102430-PA.
- He, X., Santoso, R., and Hoteit, H. 2020. Application of Machine-Learning to Construct Equivalent Continuum Models from High-Resolution Discrete-Fracture Models. Paper presented at the International Petroleum Technology Conference, Dhahran, Kingdom of Saudi Arabia, 13–15 January. IPTC-20040-MS. https://doi.org/10.2523/iptc-20040-ms.
- He, X., Alsinan, M., Zhang, Z. et al. 2022a. Micro-Continuum Approach for Modeling Coupled Flow and Geomechanical Processes in Fractured Rocks. Paper presented at the SPE Annual Technical Conference and Exhibition, Houston, Texas, USA, 3–5 October. SPE-210453-MS. https://doi.org/10.2118/210453-MS.
- He, X., Qiao, T., Alsinan, M. et al. 2021b. Discrete Fracture Model for Hydro-Mechanical Coupling in Fractured Reservoirs. Paper presented at the Abu Dhabi International Petroleum Exhibition & Conference, Abu Dhabi, UAE, 15–18 November. SPE-208039-MS. https://doi.org/10.2118/208039-MS.
- He, X., Qiao, T., Santoso, R. et al. 2021c. Gas Injection Optimization Under Uncertainty in Subsurface Reservoirs: An Integrated Machine Learning-Assisted Workflow. Paper presented at the ARMA/DGS/SEG 2nd International Geomechanics Symposium, Virtual, 1–4 November. ARMA-IGS21-009.
- He, X., Sinan, M., Kwak, H. et al. 2021a. A Corrected Cubic Law for Single-Phase Laminar Flow through Rough-Walled Fractures. *Adv Water Resour* 154: 103984. https://doi.org/10.1016/j.advwatres.2021.103984.
- He, X., Zhang, Z., Alsinan, M. et al. 2022b. Uncertainty and Sensitivity Analysis of Multi-Phase Flow in Fractured Rocks: A Pore-To-Field Scale Investigation. Paper presented at the SPE Annual Technical Conference and Exhibition, Houston, Texas, USA, 3–5 October. SPE-210131-MS. https://doi.org/10.2118/210131-MS.
- He, X., Zhu, W., AlSinan, M. et al. 2022c. CO2 Storage Capacity Prediction In Deep Saline Aquifers: Uncertainty and Global Sensitivity Analysis. Paper presented at the International Petroleum Technology Conference, Riyadh, Saudi Arabia, 21–23 February. https://doi.org/10.2523/IPTC-22463-MS.
- He, X., Zhu, W., Santoso, R. et al. 2021d. Fracture Permeability Estimation Under Complex Physics: A Data-Driven Model Using Machine Learning. Paper presented at the SPE Annual Technical Conference and Exhibition, Dubai, UAE, 21–23 September. SPE-206352-MS. https://doi.org/10.2118/206352-MS.
- He, X., Zhu, W., Santoso, R. et al. 2021e. CO2 Leakage Rate Forecasting Using Optimized Deep Learning. Paper presented at the SPE Annual Technical Conference and Exhibition, Dubai, UAE, 21–23 September. SPE-206222-MS. https://doi.org/10.2118/206222-MS.
- Hochreiter, S. and Schmidhuber, J. 1997. Long Short-Term Memory. *Neural Comput* 9 (8): 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.
- Hoffman, B. T. and Caers, J. 2004. History Matching Under Geological Control: Application to a North Sea Reservoir. In *Geostatistics Banff*, 1067–1076. Dordrecht, The Netherlands: Springer. https://doi.org/10.1007/978-1-4020-3610-1_111.
- Hoteit, H. and Firoozabadi, A. 2008a. An Efficient Numerical Model for Incompressible Two-Phase Flow in Fractured Media. *Adv Water Resour* 31 (6): 891–905. https://doi.org/10.1016/j.advwatres.2008.02.004.
- Hoteit, H. and Firoozabadi, A. 2008b. Numerical Modeling of Two-Phase Flow in Heterogeneous Permeable Media with Different Capillarity Pressures. *Adv Water Resour* 31 (1): 56–73. https://doi.org/10.1016/j.advwatres.2007.06.006.
- Hoteit, H., Fahs, M., and Soltanian, M. R. 2019. Assessment of CO2 Injectivity During Sequestration in Depleted Gas Reservoirs. *Geosciences* 9 (5): 199. https://doi.org/10.3390/geosciences9050199.
- Hoteit, H., He, X., Yan, B. et al. 2021. Optimization and Uncertainty Quantification Model for Time-Continuous Geothermal Energy Extraction Undergoing Re-Injection. ArXiv Preprint ArXiv:2112.05544.
- Hu, L. Y. 2000. Gradual Deformation and Iterative Calibration of Gaussian-Related Stochastic Models. *Math Geol* 32 (1): 87–108. https://doi.org/10.1023/A:1007506918588.
- Hu, L. Y., Blanc, G., and Noetinger, B. 2001. Gradual Deformation and Iterative Calibration of Sequential Stochastic Simulations. *Math Geol* 33 (4): 475–489. https://doi.org/10.1023/A:1011088913233.
- Jagalur-Mohan, J., Jha, B., Wang, Z. et al. 2018. Inferring Fault Frictional and Reservoir Hydraulic Properties From Injection-Induced Seismicity. *Geophys Res Lett* 45 (3): 1313–1320. https://doi.org/10.1002/2017GL075925.
- Jiang, S. and Durlofsky, L. J. 2020. Data-Space Inversion Using a Recurrent Autoencoder for Time-Series Parameterization. *Comput Geosci* 25 (1): 411–432. https://doi.org/10.1007/s10596-020-10014-1.
- Kandel, I. and Castelli, M. 2020. The Effect of Batch Size on the Generalizability of the Convolutional Neural Networks on a Histopathology Dataset. *ICT Express* 6 (4): 312–315. https://doi.org/10.1016/j.icte.2020.04.010.
- Katterbauer, K., Hoteit, I., and Sun, S. 2015. History Matching of Electromagnetically Heated Reservoirs Incorporating Full-Wavefield Seismic and Electromagnetic Imaging. *SPE J.* 20 (5): 923–941. SPE-173896-PA. https://doi.org/10.2118/173896-PA.
- Kitanidis, P. K. 1995. Quasi-Linear Geostatistical Theory for Inversing. *Water Resour Res* 31 (10): 2411–2419. https://doi.org/10.1029/95WR01945.
- Lee, T., Kravaris, C., and Seinfeld, J. H. 1986. History Matching by Spline Approximation and Regularization in Single-Phase Areal Reservoirs. *SPE Res Eng* 1 (5): 521–534. SPE-13931-PA. https://doi.org/10.2118/13931-PA.
- Li, B., Bhark, E. W., Billiter, T. C. et al. 2019. Best Practices of Assisted History Matching Using Design of Experiments. *SPE J.* 24 (4): 1435–1451. SPE-191699-PA. https://doi.org/10.2118/191699-PA.
- Link, W. A. and Eaton, M. J. 2012. On Thinning of Chains in MCMC. *Methods Ecol Evol* 3 (1): 112–115. https://doi.org/10.1111/j.2041-210X.2011.00131.x.
- Liu, Y. and Durlofsky, L. J. 2020. Multilevel Strategies and Geological Parameterizations for History Matching Complex Reservoir Models. *SPE J.* 25 (1): 081–104. SPE-193895-PA. https://doi.org/10.2118/193895-PA.
- Mahendran, N., Wang, Z., Hamze, F. et al. 2011. Bayesian Optimization for Adaptive MCMC. ArXiv:1110.6497.
- Makhlouf, E. M., Chen, W. H., Wasserman, M. L. et al. 1993. A General History Matching Algorithm for Three-Phase, Three-Dimensional Petroleum Reservoirs. *SPE Adv Technol Ser* 1 (2): 83–92. SPE-20383-PA. https://doi.org/10.2118/20383-PA.
- Marelli, S. and Sudret, B. 2014. UQLab: A Framework for Uncertainty Quantification in MATLAB. Paper presented at the Second International Conference on Vulnerability and Risk Analysis and Management (ICVRAM) and the Sixth International Symposium on Uncertainty, Modeling, and Analysis (ISUMA), Liverpool, UK, 13–16 July. https://doi.org/10.1061/9780784413609.257.
- Martino, L. 2018. A Review of Multiple Try MCMC Algorithms for Signal Processing. *Digit Signal Process* 75: 134–152. https://doi.org/10.1016/j.dsp.2018.01.004.
- Marzouk, Y. M., Najm, H. N., and Rahn, L. A. 2007. Stochastic Spectral Methods for Efficient Bayesian Solution of Inverse Problems. *J Comput Phys* 224 (2): 560–586. https://doi.org/10.1016/j.jcp.2006.10.010.
- Moore, C. C. 2015. Ergodic Theorem, Ergodic Theory, and Statistical Mechanics. *Proc Natl Acad Sci USA* 112 (7): 1907–1911. https://doi.org/10.1073/pnas.1421798112.
- Naik, P., Pandita, P., Aramideh, S. et al. 2019. Bayesian Model Calibration and Optimization of Surfactant-Polymer Flooding. *Comput Geosci* 23 (5): 981–996. https://doi.org/10.1007/s10596-019-09858-z.
- Oliver, D. S. and Chen, Y. 2011. Recent Progress on Reservoir History Matching: A Review. *Comput Geosci* 15 (1): 185–221. https://doi.org/10.1007/s10596-010-9194-2.
- Oliver, D. S., He, N., and Reynolds, A. C. 1996. Conditioning Permeability Fields to Pressure Data. Paper presented at the 5th European Conference on Mathematics of Oil Recovery, Leoben, Austria.
- Omar, A., Addassi, M., Vahrenkamp, V. et al. 2021. Co-Optimization of CO2 Storage and Enhanced Gas Recovery Using Carbonated Water and Supercritical CO2. *Energies* 14 (22): 7495. https://doi.org/10.3390/en14227495.
- Pal, N., Zhang, X., Ali, M. et al. 2022. Carbon Dioxide Thickening: A Review of Technological Aspects, Advances and Challenges for Oilfield Application. *Fuel* 315: 122947. https://doi.org/10.1016/j.fuel.2021.122947.
- Pascanu, R., Gulcehre, C., Cho, K. et al. 2013. How to Construct Deep Recurrent Neural Networks. ArXiv:1312.6026.
- Rasmussen, C. E. and Williams, C. K. I. 2006. *Gaussian Processes for Machine Learning*. Cambridge, Massachusetts: MIT Press.
- Romero, C. E. and Carter, J. N. 2001. Using Genetic Algorithms for Reservoir Characterisation. *J Pet Sci Eng* 31 (2–4): 113–123. https://doi.org/10.1016/S0920-4105(01)00124-3.
- Rwechungura, R., Dadashpour, M., and Kleppe, J. 2011. Advanced History Matching Techniques Reviewed. Paper presented at the SPE Middle East Oil and Gas Show and Conference, Manama, Bahrain, 25–28 September. SPE-142497-MS. https://doi.org/10.2118/142497-MS.
- Saksman, E. and Vihola, M. 2010. On the Ergodicity of the Adaptive Metropolis Algorithm on Unbounded Domains. *Ann Appl Probab* 20 (6): 2178–2203. https://doi.org/10.1214/10-AAP682.
- Santoso, R., He, X., and Hoteit, H. 2019a. Application of Machine-Learning to Construct Simulation Models from High-Resolution Fractured Formation. Paper presented at the Abu Dhabi International Petroleum Exhibition & Conference, Abu Dhabi, UAE, 11–14 November. SPE-197439-MS. https://doi.org/10.2118/197439-MS.
- Santoso, R., Hoteit, H., and Vahrenkamp, V. 2019b. Optimization of Energy Recovery from Geothermal Reservoirs Undergoing Re-Injection: Conceptual Application in Saudi Arabia. Paper presented at the SPE Middle East Oil and Gas Show and Conference, Manama, Bahrain, 18–21 March. SPE-195155-MS. https://doi.org/10.2118/195155-MS.
- Santoso, R., Torrealba, V., and Hoteit, H. 2020. Investigation of an Improved Polymer Flooding Scheme by Compositionally-Tuned Slugs. *Processes* 8 (2): 197. https://doi.org/10.3390/pr8020197.
- Santoso, R., He, X., Alsinan, M. et al. 2021a. Multi-Fidelity Bayesian Approach for History Matching in Reservoir Simulation. Paper presented at the SPE Middle East Oil & Gas Show and Conference, Manama, Bahrain, 7–9 March. SPE-204652-MS. https://doi.org/10.2118/204652-MS.
- Santoso, R., He, X., Alsinan, M. et al. 2021b. Bayesian Long-Short Term Memory for History Matching in Reservoir Simulations. Paper presented at the SPE Reservoir Simulation Conference, On-Demand, 26 October. SPE-203976-MS. https://doi.org/10.2118/203976-MS.
- Schulze-Riegert, R. W., Haase, O., and Nekrassov, A. 2003. Combined Global and Local Optimization Techniques Applied to History Matching. Paper presented at the SPE Reservoir Simulation Symposium, Houston, Texas, USA, 3–5 February. SPE-79668-MS. https://doi.org/10.2118/79668-MS.
- Schulze-Riegert, R. W., Axmann, J. K., Haase, O. et al. 2002. Evolutionary Algorithms Applied to History Matching of Complex Reservoirs. *SPE Res Eval & Eng* 5 (2): 163–173. SPE-77301-PA. https://doi.org/10.2118/77301-PA.
- Shabbeer, B. S. H., Ram, D. S., Pulabaigari, V. et al. 2019. Impact of Fully Connected Layers on Performance of Convolutional Neural Networks for Image Classification. *Neurocomputing* 378: 112–119. https://doi.org/10.1016/j.neucom.2019.10.008.
- Shi, X., Chen, Z., Wang, H. et al. 2015. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. Paper presented at the Neural Information Processing Systems, Montreal, Quebec, Canada, 7–12 December.
- Suzuki, S., Caumon, G., and Caers, J. 2008. Dynamic Data Integration for Structural Modeling: Model Screening Approach Using a Distance-Based Model Parameterization. *Comput Geosci* 12 (1): 105–119. https://doi.org/10.1007/s10596-007-9063-9.
- Tang, M., Liu, Y., and Durlofsky, L. J. 2020. A Deep-Learning-Based Surrogate Model for Data Assimilation in Dynamic Subsurface Flow Problems. *J Comput Phys* 413: 109456. https://doi.org/10.1016/j.jcp.2020.109456.
- Ţene, M., Al Kobaisi, M. S., and Hajibeygi, H. 2016. Algebraic Multiscale Method for Flow in Heterogeneous Porous Media with Embedded Discrete Fractures (F-AMS). *J Comput Phys* 321: 819–845. https://doi.org/10.1016/j.jcp.2016.06.012.
- Vefring, E. H., Nygaard, G., Lorentzen, R. J. et al. 2006. Reservoir Characterization During Underbalanced Drilling (UBD): Methodology and Active Tests. *SPE J.* 11 (2): 181–192. SPE-81634-PA. https://doi.org/10.2118/81634-PA.
- Wagner, P. R., Nagel, J., Marelli, S. et al. 2019. *UQLab User Manual – Bayesian Inversion for Model Calibration and Validation*. Zurich, Switzerland: Chair of Risk, Safety and Uncertainty Quantification.
- Williams, M. A., Keating, J. F., and Barghouty, M. F. 1998. The Stratigraphic Method: A Structured Approach to History-Matching Complex Simulation Models. *SPE Res Eval & Eng* 1 (2): 169–176. SPE-38014-PA. https://doi.org/10.2118/38014-PA.
- Wilson, J. T., Hutter, F., and Deisenroth, M. P. 2018. Maximizing Acquisition Functions for Bayesian Optimization. Paper presented at the Neural Information Processing Systems, Montréal, Canada, 3 August.
- Winkler, R. L. 1967. The Assessment of Prior Distributions in Bayesian Analysis. *J Am Stat Assoc* 62 (319): 776–800. https://doi.org/10.1080/01621459.1967.10500894.
- Yang, P. H. and Watson, A. T. 1988. Automatic History Matching With Variable-Metric Methods. *SPE Res Eng* 3 (3): 995–1001. SPE-16977-PA. https://doi.org/10.2118/16977-PA.
- Zhang, X. and Curtis, A. 2020. Seismic Tomography Using Variational Inference Methods. *J Geophys Res Solid Earth* 125 (4): 4. https://doi.org/10.1029/2019JB018589.
- Zhang, Y. and Oliver, D. S. 2011. History Matching Using the Ensemble Kalman Filter With Multiscale Parameterization: A Field Case Study. *SPE J.* 16 (2): 307–317. SPE-118879-PA. https://doi.org/10.2118/118879-PA.
- Zhang, F., Reynolds, A. C., and Oliver, D. S. 2003. An Initial Guess for the Levenberg-Marquardt Algorithm for Conditioning a Stochastic Channel to Pressure Data. *Math Geol* 35 (1): 67–88. https://doi.org/10.1023/A:1022317229206.
- Zhang, Z., He, X., AlSinan, M. et al. 2022a. Deep Learning Model for CO2 Leakage Detection Using Pressure Measurements. Paper presented at the SPE Annual Technical Conference and Exhibition, Houston, Texas, USA, 3–5 October. SPE-209959-MS. https://doi.org/10.2118/209959-MS.
- Zhang, Z., Li, Y., AlSinan, M. et al. 2022b. Multiscale Carbonate Rock Reconstruction Using a Hybrid WGAN-GP and Super-Resolution. Paper presented at the SPE Annual Technical Conference and Exhibition, Houston, Texas, USA, 3–5 October. SPE-210461-MS. https://doi.org/10.2118/210461-MS.
