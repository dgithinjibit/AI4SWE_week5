# Part 3: Critical Thinking – Ethics, Bias, and Trade-offs in Healthcare AI

This section explores the deeper implications of deploying AI in high-stakes domains like healthcare, focusing on ethical risks, fairness strategies, and practical constraints that shape model design and deployment decisions.

---

## Ethics & Bias

### How Biased Training Data Affects Patient Outcomes

Biased training data can perpetuate and even amplify historical inequities in healthcare. For example:

- If minority patients have been **systematically discharged earlier** due to implicit bias or bed pressure,
- And if they subsequently experience higher readmission rates due to **lack of access to follow-up care, transportation, or medication affordability** (social determinants of health),
- Then the model learns to associate **race, zip code, or insurance type** with higher readmission risk — not because of biological differences, but because of structural injustice.

As a result:
- The model will **over-predict risk for marginalized groups**, leading to excessive monitoring or paternalistic interventions.
- Conversely, it may **under-predict risk for wealthier patients** who historically received prolonged care, missing genuine threats.
- This creates a **feedback loop**: biased predictions → unequal treatment → reinforced disparities → more biased future data.

Ultimately, such a system erodes trust, violates principles of equity, and leads to **unfair patient outcomes**, where care allocation is driven by systemic bias rather than clinical need.

---

### Strategy to Mitigate Bias

**Approach: Disparate Impact Analysis + Post-Processing Re-calibration (Equalized Odds)**

To break the cycle of algorithmic bias, we implement a two-step fairness strategy:

1. **Disparate Impact Analysis:**  
   After initial model training, evaluate performance across protected groups (e.g., by race, insurance type, or socioeconomic status). Key metrics to compare:
   - **False Positive Rate (FPR):** % of healthy patients incorrectly flagged as high-risk
   - **False Negative Rate (FNR):** % of truly at-risk patients missed by the model

2. **Post-Processing Threshold Adjustment (Equalized Odds):**  
   If significant disparities are found (e.g., FPR is 40% for Medicaid patients vs. 15% for Private insurance), apply **group-specific decision thresholds**:
   - Raise the classification threshold for groups with lower baseline risk
   - Lower it slightly for historically disadvantaged groups (if justified clinically)

The goal is to achieve **parity in both FPR and FNR** across groups — ensuring the model is equally accurate and fair regardless of patient background.

> ✅ Why this works: It decouples prediction from proxy variables while preserving overall utility. Clinicians still get reliable risk scores, but now with **equitable error rates**.

---

## Trade-offs in Model Design

### Interpretability vs. Accuracy in Healthcare

| Dimension | High-Accuracy Models (e.g., Deep Learning) | Interpretable Models (e.g., Logistic Regression) |
|--------|--------------------------------------------|--------------------------------------------------|
| **Accuracy** | Higher – captures complex, non-linear patterns | Slightly lower – assumes linear relationships |
| **Interpretability** | Low ("black box") – hard to explain predictions | High – coefficients show feature impact clearly |
| **Clinical Trust** | Low – doctors hesitate to act on unexplainable alerts | High – transparent logic supports shared decision-making |
| **Regulatory Risk** | High – difficult to audit or justify under HIPAA/AI regulations | Low – easy to validate and document |

**Conclusion:**  
In healthcare, **interpretability often outweighs marginal gains in accuracy**. A doctor needs to know *why* a patient is flagged — was it due to comorbidities? Poor lab trends? Socioeconomic risk? — to take meaningful action.

Choosing **Logistic Regression** over a neural network ensures:
- Predictions can be explained using simple rules
- Coefficients align with clinical intuition
- The model supports **accountable, defensible care**

Thus, sacrificing a few percentage points in AUC for full transparency is not just acceptable — it's ethically and operationally necessary.

---

### Impact of Limited Computational Resources on Model Choice

Hospitals, especially in resource-constrained settings, often operate with outdated infrastructure or limited cloud budgets. This has direct consequences on model selection:

#### Constraints Imposed:
- No GPU availability
- Memory limits (<4GB RAM per service)
- Need for low-latency inference (<500ms response time)
- Serverless function timeouts (e.g., AWS Lambda max 15 minutes)

#### Resulting Model Priorities:
1. **Efficiency Over Complexity:**  
   Rule out deep learning and large ensembles (e.g., XGBoost with 1000 trees).
2. **Fast Inference & Low Footprint:**  
   Favor lightweight models like **Logistic Regression**, **Decision Trees**, or **Random Forest** (with shallow depth).
3. **Scalability in Serverless Environments:**  
   Smaller models start faster, reducing cold-start delays in serverless APIs — critical for real-time EHR integration.

#### Strategic Alignment:
This constraint aligns with my focus on **robust, scalable, serverless architectures**. By choosing efficient models:
- We reduce operational costs
- Improve reliability in low-resource environments
- Enable deployment across urban and potentially rural clinics
- Maintain alignment with clean, maintainable code practices

> 💡 **Key Insight:** Simplicity isn't a compromise — it's a feature when building sustainable, equitable AI systems in real-world healthcare settings.

---
