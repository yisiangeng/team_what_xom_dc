**FAQ – DataWorks Challenge 2026**

**1. What is the objective of this challenge?**

The objective is to identify and optimize the uncertain dynamic parameters using the information provided in the dataset, develop models that reasonably reproduce historical field performance, and generate credible production forecasts.

**2. Are participants expected to predict future operational changes?**

No. Participants should assume that future operating conditions remain consistent with those represented in the provided dataset. The challenge focuses on uncertainty characterization and forecasting under the stated assumptions.

**3. Is additional data available beyond the provided dataset?**

No. Participants should use only the data provided as part of the challenge package.

**4. Is history matching required?**

Yes. Participants are expected to calibrate their models against the available historical production data before generating forecasts.

**5. What constitutes a good history match?**

A good history match should:

- Reproduce the overall production trends.
- Capture key reservoir and production behavior.
- Provide a reasonable agreement with observed field performance.
- Remain physically realistic.

In general, simulation models should follow the observed production trends and ultimately achieve a reasonable match to the field data.

**6. Should participants prioritize a perfect match at every data point?**

No. The challenge focuses on obtaining representative reservoir behavior rather than achieving a perfect point-by-point match.

**7. How many forecast cases should be submitted?**

Participants should submit the forecast profiles corresponding to their selected 10 best cases.

**8. How should the 10 best cases be selected?**

The selected cases should be the realizations that most closely represent the historical field performance while remaining technically defensible.

**9. Should the selected cases represent uncertainty diversity or forecast accuracy?**

The primary objective is to identify cases that best represent the historical field behavior. While uncertainty representation is important, selection should be driven by history-match quality and forecasting credibility.

**10. Should the 10 best forecast cases be selected from the training, validation, or test cases only?**

No. The 10 best forecast cases should be selected from the full population of generated realizations.

**11. Is forecast evaluation performed at the field level or well level?**

Forecast evaluation is performed at the field level.

**12. Why do some production profiles show zero production during the first time steps?**

This is intentional. The field had not yet started producing during the initial periods shown in the dataset; therefore, zero production values at the beginning of the production history are expected.

**13. Should participants treat the early zero-rate periods as missing data?**

No. These values are valid observations and should be interpreted as part of the field production history.

**14. The kickoff presentation mentioned a 20-year forecast period, but the spreadsheet initially displayed only 10 years. Which is correct?**

The intended forecast horizon is 20 years. The new template is now available on the starter pack on the website.

**15. Is there a single correct answer to the challenge?**

No. Multiple solutions may produce acceptable forecasts. Participants are encouraged to develop technically sound workflows and support their decisions with appropriate engineering rationale.

**Key Takeaway**

The challenge is not simply about obtaining the lowest error metric. It is designed to evaluate a participant's ability to integrate technical knowledge, engineering judgment, uncertainty management, and forecasting skills to develop a robust and defensible prediction of future field performance.
