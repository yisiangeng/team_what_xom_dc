# DataWorks Challenge 2026 (DWC26) — Use Case Briefing

> Source: *DataWorks 2026 – Kick Off* (slides 21–35). Organiser branding on every slide: ExxonMobil.
> Wording is taken from the slides. Layout and readability have been tidied. Notes in *italics* or marked **[Image]** are my interpretation of visuals, not slide text.

## Contents

- [Slide 21 — Use Case Briefing (section divider)](#slide-21--use-case-briefing-section-divider)
- [Slide 22 — Title: Machine Learning-Assisted History Matching](#slide-22--title)
- [Slide 23 — DataWorks Challenge (overview graphic)](#slide-23--dataworks-challenge-overview-graphic)
- [Slide 24 — Reservoir Model and Rock Property](#slide-24--reservoir-model-and-rock-property)
- [Slide 25 — Problem Statement](#slide-25--problem-statement)
- [Slide 26 — Objectives](#slide-26--objectives)
- [Slide 27 — Scene Setting](#slide-27--scene-setting)
- [Slide 28 — Dataset Description](#slide-28--dataset-description)
- [Slide 29 — Dataset Analysis](#slide-29--dataset-analysis)
- [Slide 30 — Suggested Workflow](#slide-30--suggested-workflow)
- [Slide 31 — Results](#slide-31--results)
- [Slide 32 — References](#slide-32--references)
- [Slide 33 — Deliverables & Evaluation (section divider)](#slide-33--deliverables--evaluation-section-divider)
- [Slide 34 — Deliverables](#slide-34--deliverables)
- [Slide 35 — Evaluation](#slide-35--evaluation)

---

## Slide 21 — Use Case Briefing (section divider)

**Use Case Briefing**

*Section divider slide. The only other text is the tag "DWC26".*

---

## Slide 22 — Title

**DataWorks Challenge 2026**

**Machine Learning-Assisted History Matching and the Challenge of Production Forecasting**

---

## Slide 23 — DataWorks Challenge (overview graphic)

**Title:** DataWorks Challenge

This slide has no body text, only pictures. **[Image]** It reads as a left-to-right flow in three stages joined by two blue arrows:

1. **Inputs (left):** a stack of subsurface-data pictures:
   - a cut-away block of seabed and rock layers with a platform on top
   - core or rock-sample photos
   - a canyon or rock-outcrop photo
   - well-log curves
   - a dish of crude oil
   - a capillary-pressure chart (curves of different colours)
2. **Reservoir model (middle):** a 3D colour-mapped reservoir grid. This is probably the simulation model.
3. **Outputs (right):** a 2×3 grid of production-forecast charts, each with a bundle of coloured curves and black dots (probably the observed data). A photo of coins stacked with seedlings growing from them sits in the middle of the grid. I take it to mean production forecasts turn into money or value.

The chart titles that can be made out are:

- Field, Gas production cumulative
- Field, Gas production rate
- Field, Oil production cumulative
- Field, Oil production rate
- Field, Water production cumulative
- Field, Water production rate

The overall message appears to be: subsurface data → reservoir simulation model → production forecasts → business value. The slide does not say this in words.

---

## Slide 24 — Reservoir Model and Rock Property

**Title:** DataWorks Challenge

### Reservoir Model

A reservoir simulation model is a mathematical representation of the reservoir that is used to reproduce actual reservoir behavior and predict field performance under different development scenarios.

It is represented by thousands to millions grid cells which are assigned with rock and fluid properties.

> **[Image, left]** A 3D colour-mapped reservoir model, the same one as on slide 23.

### Rock Property

| Property | Definition |
|---|---|
| **Porosity** | The fraction of a rock's total volume that consists of pore spaces capable of storing fluids. |
| **Permeability** | The ability of a reservoir rock to allow fluids to flow through its interconnected pore spaces. |
| **Transmissibility** | A measure of the ease with which fluids flow between reservoir grid cells, determined by rock properties, geometry, and fluid mobility |

> **[Image, middle]** An irregular, curved reservoir body in colour, with a zoomed-in block of a few grid cells below it. One red cell is labelled **k = 250 mD, Ø = 23%, Sw = 50%**. That is permeability, porosity and water saturation for a single grid cell.

### The governing equations

Reservoir simulation models solve partial differential equations (PDEs) derived from mass-balance principles to predict fluid flow, pressure distribution, and production performance within the reservoir.

> **[Image, right]** A set of equations, shown on a dark background:
> - There are three flow-equation lines, for oil, water and gas, each with a "+ source" term. Below them are the constraints S<sub>o</sub> + S<sub>w</sub> + S<sub>g</sub> = 1, P<sub>cow</sub> = P<sub>o</sub> − P<sub>w</sub> and P<sub>cog</sub> = P<sub>g</sub> − P<sub>o</sub>.
> - Colour-coded callouts point to groups of terms:
>   - **"Geologic" properties:** φ, k = f(P, T)
>   - **Fluid properties:** μ, ρ, B, R<sub>s</sub> = f(P, T)
>   - **Displacement properties, depend on rock-fluid interaction:** k<sub>r</sub> and P<sub>c</sub> = f(S)
>
> I can read the structure and the callouts. I have not checked each term of the equations.

---

## Slide 25 — Problem Statement

**Title:** DataWorks Challenge: Problem Statement

1. Reservoir engineers use **history matching** to align reservoir simulations with historical production data. The resulting models serve as the foundation for production forecasts that drive critical investment and business decisions.

2. This process is traditionally **time-consuming and computationally expensive**, requiring multiple runs of numerical simulators.

3. In this challenge, participants are provided with a **synthetic reservoir dataset** containing:
   - A set of reservoir model realizations with varying subsurface parameters
   - Corresponding simulated production data (oil rate, water rate, pressure)
   - A separate set of **observed production data** representing the "true" field behavior

*Each point has a small icon: a database, a maze, and a chip.*

---

## Slide 26 — Objectives

**Title:** DataWorks Challenge: Objectives

1. Build a predictive model (proxy) that maps reservoir parameters to production responses
2. Identify the optimal set of reservoir parameters that **minimize the mismatch** between simulated and observed production data
3. Generate a history-matched model that reproduces the observed field performance
4. Forecast field production over the next 20 years

*Each point has a small icon: database, gears, DNA helix, and a rising line chart.*

---

## Slide 27 — Scene Setting

**Title:** DataWorks Challenge: Scene Setting

The slide splits the work into two columns: what **ExxonMobil** provides and what the **Students** do.

### ExxonMobil

| Step | Question | Detail |
|---|---|---|
| **1. Decode on Problem** | (What are we modeling?) | **Problem:** High Fidelity Physics Model Reduction |
| **2. Curate Data** | (What DATA will inform the model?) | **Data:** Simulation Input and Output Dataset generated from running physics-based model |

### Students

| Step | Question |
|---|---|
| **3. Design an Architecture** | (RNN, Autoencoder, DMD?) |
| **4. Craft a Loss Function** | (What models are "GOOD"?) |
| **5. Employ Optimization** | (What Algorithms to Train Model?) |

---

## Slide 28 — Dataset Description

**Title:** DataWorks Challenge: Dataset Description

The slide has three panels: **Introduction**, **Field Oil and Gas Production** and **Simulation Cases**.

### Introduction

*Removed from this file. The slide shows two tables: (a) the data dictionary of variables and (b) the first 6 rows of the case-parameter table. Both are subsets of `dataset/01 Introduction.xlsx`: the wording and values match, and the workbook holds the full dictionary and all 100 cases. The complete description now lives in `dev_notes/data_desc.md`.*

### Field Oil and Gas Production

**10 years of production data**

**[Image]** Six small charts in a 2×3 grid. They show the single observed history as black dots:

- Gas production cumulative
- Gas production rate
- Oil production cumulative
- Oil production rate
- Water production cumulative
- Water production rate

### Simulation Cases

- **70** Train cases
- **15** Validation cases
- **15** Test cases

**[Image]** The same six charts, now with many coloured curves, one per simulated case. They fan out into a wide band over the 10 years.

---

## Slide 29 — Dataset Analysis

**Title:** DataWorks Challenge: Dataset Analysis

**Caption:** *Subsurface uncertainty reflected in a wide range of production forecast*

**[Image]** A large chart titled **Field, Oil production cumulative** (y-axis: oil production cumulative [STB], up to about 1.2E+08; x-axis: Date, Jan 2000 to Jan 2030). It shows a dense bundle of coloured curves for all the cases. They are identical early on, then spread into a wide band that flattens out toward 2030.

Three smaller insets show how single parameters change the curves:

| Inset | Label on slide | What the inset shows |
|---|---|---|
| Top-left | **Increasing Permeability** | Field oil production cumulative: curves split from a common early trend, ordered by permeability |
| Bottom-left | **Increasing Aquifer Size/Porosity** | Field water production cumulative: a stack of curves spread out, ordered by aquifer size/porosity |
| Bottom-right | **Zero Fault Transmissibility** | Field oil production cumulative: two curves, one red and one green, with the green slightly higher |

The direction of the ordering within each inset (for example, which curve is the highest permeability) is not labelled on the slide. I have not guessed it.

---

## Slide 30 — Suggested Workflow

**Title:** DataWorks Challenge: Suggested Workflow

A five-step workflow, shown left to right as icons:

1. **Data Preparation and Analysis** *(bar chart icon)*
2. **ML Proxy Model Training** *(person icon)*
3. **ML Prediction** *(head with gears icon)*
4. **Optimization (History Matching)** *(maze icon)*
5. **Forecast Production** *(rising line chart icon)*

> **Remember:**
> - Goal of the model is to make prediction on production forecast so that we can evaluate if the field will make money
> - Rock properties are assumed to be static (i.e. will not change with time)

---

## Slide 31 — Results

**Title:** DataWorks Challenge: Results

Three panels, left to right:

1. **Provided Curves**
   **[Image]** A chart titled *Field, Gas production cumulative* (x-axis about Jan 1998 to Jan 2008). It shows many coloured simulated curves. A red line with black dots (the observed history) runs along the start of the bundle.

2. **Optimized Curves**
   **[Image]** The same chart, but the curves are now squeezed into a narrow band that matches the history. A vertical black dashed line sits at roughly the end of the history, and the x-axis extends further to the right, to about Jan 2018. An icon of two people with a question mark sits next to the question below.
   > How are the production curves expected to evolve over time?

3. **Forecast template**
   **[Image]** A spreadsheet template with these columns: **Date | Gas production cumulative [MSCF] | Oil production cumulative [STB] | Water production cumulative [STB]**. The Date column has quarterly dates starting at 04/01/2008. The value columns are empty. Over it is the text:
   > Pick the best 10 forecast cases

*I read this slide as the expected output: after history matching, the curves narrow to fit the observed data, and participants then submit forecasts for the best 10 cases in the template. The slide does not spell out this reading. The template is also cropped at 07/01/2017, so I cannot confirm the full forecast period. Slide 26 says the forecast covers "the next 20 years".*

---

## Slide 32 — References

**Title:** DataWorks Challenge: References

1. Hu Huang et al. "A Deep-Learning-Based Graph Neural Network-Long-Short-Term Memory Model for Reservoir Simulation and Optimization with Varying Well Controls, SPE 2023
2. Zhen Zhang et al, "Robust Method for Reservoir Simulation History Matching Using Bayesian Inversion and Long-Short-Term Memory Network-Based Proxy", SPE 2023

---

## Slide 33 — Deliverables & Evaluation (section divider)

**Deliverables & Evaluation**

*Section divider slide. The only other text is the tag "DWC26".*

---

## Slide 34 — Deliverables

**Title:** Deliverables

| # | Deliverable | Submit by |
|---|---|---|
| 1 | **Code Repository** | **16 Oct** |
| 2 | **Documentation** (in PowerPoint) | **20 Oct** |
| 3 | **Power BI Visualization** | **20 Oct** |
| 4 | **Video** | **20 Oct** |

*On the slide, a brace under deliverable 1 reads "Submit by 16 Oct". A second, wider brace under deliverables 2–4 reads "Submit by 20 Oct".*

---

## Slide 35 — Evaluation

**Title:** Evaluation

Three criteria, each in a box with a green tick:

1. **All 4 deliverables submitted on time**
   *(Icons for the four deliverables: code, document, chart, video.)*

2. **Accuracy of Model Output**
   *based on output results and code review*

3. **Submission Quality**, covering these areas:
   - Business Understanding
   - Technical Understanding
   - Communication & Storytelling
   - Insights
   - Innovation
   - Collaboration
   - Bonus

*The slide gives no weightings or scoring, only the criteria.*

---

## Things to watch for

- **Slide 29 inset direction.** The slide does not say which way each parameter is ordered (for example, highest permeability on top or bottom).
- **Deadlines.** Code Repository is due 16 Oct. Documentation, Power BI and Video are due 20 Oct.
