# Part 2: Case Study Application – Predicting 30-Day Hospital Readmission Risk

This document outlines the end-to-end design of an AI system to predict patient readmission risk within 30 days of discharge, focusing on clinical impact, ethical integrity, and deployment feasibility in a regulated healthcare environment.

---

## Problem Scope

### Problem Definition  
Predict which patients are at high risk of being readmitted to the hospital within 30 days of discharge using historical and clinical data. The goal is to enable proactive post-discharge interventions that improve outcomes and reduce unnecessary hospitalizations.

### Objectives
1. **Improve Patient Outcomes:** Identify high-risk individuals for targeted follow-up care (e.g., telehealth check-ins, home visits) to prevent complications.
2. **Reduce Healthcare Costs:** Lower avoidable readmissions, which incur financial penalties under value-based care models.
3. **Optimize Resource Allocation:** Direct limited care coordination staff toward patients with the highest predicted risk, improving efficiency and impact.

### Stakeholders
- **Patients:** Benefit from personalized post-discharge support and reduced health risks.
- **Hospital Administration:** Concerned with quality metrics, operational costs, and compliance with regulatory standards (e.g., CMS readmission benchmarks).

---

## Data Strategy

### Proposed Data Sources
1. **Electronic Health Records (EHRs):**  
   - Diagnoses (ICD-10 codes), medication lists, lab results, vital signs, procedure history  
   - Length of stay, discharge disposition, comorbidities  

2. **Socioeconomic and Demographic Data (Ethically Sourced):**  
   - Age, gender, insurance type (Medicaid/Medicare/Private)  
   - Geographic location (zip code — used cautiously due to bias risk)  
   - Self-reported social determinants of health (SDOH), where available and consented  

> *Note:* All personally identifiable information (PII) and protected health information (PHI) must be handled per HIPAA regulations.

### Ethical Concerns
1. **Patient Privacy & Data Anonymization:**  
   Any use of PHI requires strict de-identification (pseudonymization or full anonymization). Access to raw data must be logged and restricted via role-based controls. Model training environments must be isolated and encrypted.

2. **Algorithmic Redlining / Bias in Treatment:**  
   Features like zip code or insurance status can act as proxies for race or income. If unaddressed, the model may unfairly target low-income or minority patients for intensive monitoring while under-servicing others. This could reinforce systemic inequities and damage trust.

> **Mitigation Approach:** Regular fairness audits, feature sensitivity analysis, and post-processing calibration (e.g., Equalized Odds) across demographic groups.

### Preprocessing Pipeline
1. **Missing Data Imputation:**  
   Use group-specific imputation (e.g., median blood pressure by diagnosis cohort) rather than global averages to preserve clinical relevance.

2. **Temporal Feature Extraction:**  
   Derive `length_of_stay` from admission and discharge timestamps. Categorize as short/medium/long based on clinical thresholds.

3. **Feature Engineering – Comorbidity Index:**  
   Calculate the **Charlson Comorbidity Index (CCI)** from ICD-10 codes. This single, clinically validated score quantifies overall disease burden and strongly predicts mortality and readmission risk.

4. **Normalization & Encoding:**  
   - Scale numeric features (e.g., age, lab values) using Min-Max scaling  
   - One-hot encode categorical variables (e.g., insurance type, primary diagnosis)

---

## Model Development

### Selected Model & Justification
**Model:** **Logistic Regression with L2 Regularization**  
**Justification:**  
In high-stakes healthcare settings, **interpretability is paramount**. Unlike black-box models (e.g., deep neural networks), Logistic Regression provides:
- Clear probability outputs (e.g., "68% chance of readmission")
- Interpretable coefficients showing direction and magnitude of each feature’s influence (e.g., “Diabetes increases log-odds by 0.4”)
- Easier validation by clinicians and auditors
- Simpler debugging and regulatory approval pathway

While slightly less accurate than ensemble methods, its transparency ensures **clinical trust**, **accountability**, and **compliance readiness**—critical for adoption.

### Confusion Matrix & Metrics (Hypothetical Test Set)

|                     | Predicted: Readmit (Positive) | Predicted: Not Readmit (Negative) |
|---------------------|-------------------------------|-----------------------------------|
| **Actual: Readmit (Positive)**   | True Positive (TP): 90        | False Negative (FN): 30           |
| **Actual: Not Readmit (Negative)** | False Positive (FP): 60       | True Negative (TN): 820           |

- Total Actual Readmissions: $90 + 30 = 120$
- Total Predicted Readmissions: $90 + 60 = 150$

#### Calculated Metrics:
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{90}{150} = \mathbf{0.60}
$$
> Of all patients flagged as high-risk, 60% actually readmitted. Indicates moderate efficiency in targeting interventions.

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{90}{120} = \mathbf{0.75}
$$
> The model successfully identified 75% of all actual readmissions. Strong coverage of true at-risk cases.

#### Interpretation:
The model achieves a reasonable balance between catching real risks (high recall) and avoiding wasted effort (moderate precision). Trade-offs can be adjusted via decision threshold tuning based on hospital priorities.

---

## Deployment Plan

### Integration Steps into Hospital System
1. **API Endpoint Creation:**  
   Wrap the trained model in a secure REST API (e.g., Flask/FastAPI microservice) hosted in a HIPAA-compliant environment. Expose endpoint: `POST /predict_risk`.

2. **EHR System Hook:**  
   Integrate with the hospital’s EHR (e.g., Epic or Cerner) via HL7/FHIR interface. Trigger prediction when a clinician finalizes a discharge order.

3. **Risk Flag Delivery:**  
   Return structured response including:
   - Readmission probability (e.g., `0.85`)
   - Top 3 contributing factors (from logistic regression coefficients)
   - Recommended action level (Low/Medium/High)

4. **Dashboard Integration:**  
   Display risk flag and insights directly in the patient’s discharge summary view for nurses and care coordinators.

---

### Ensuring Regulatory Compliance (HIPAA)
Adopt a **"Security and Privacy by Design"** framework:

- **De-Identification:** Strip all direct identifiers before model training; use tokenized patient IDs.
- **Compliant Hosting:** Deploy model backend in a HIPAA-eligible cloud environment (e.g., AWS with BAA).
- **Encryption:** Enforce TLS 1.3+ for data in transit; AES-256 encryption for data at rest.
- **Access Control:** Implement Role-Based Access Control (RBAC); only authorized roles (e.g., care team leads) can trigger predictions.
- **Audit Logging:** Log all API calls with user ID, timestamp, and purpose for accountability.

---

## Optimization: Addressing Overfitting

### Method: **L2 (Ridge) Regularization**
Apply L2 regularization to the Logistic Regression loss function:
$$
\mathcal{L} = \text{LogLoss} + \lambda \sum_{j=1}^{p} \beta_j^2
$$
Where $\lambda$ controls the strength of penalty.

**Why It Works:**
- Penalizes large coefficient values, preventing the model from over-relying on any single feature.
- Encourages smoother, more generalizable decision boundaries.
- Particularly effective when dealing with correlated features (e.g., multiple lab tests measuring kidney function).

**Impact:** Improves model robustness on unseen patient data, reducing variance and enhancing real-world performance.

---
