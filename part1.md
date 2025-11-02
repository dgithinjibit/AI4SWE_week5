# Part 1: Short Answer Questions – AI Project Design for SyncSenta

This document outlines the foundational design and planning for an AI system aimed at improving educator engagement on the **SyncSenta** platform by predicting optimal intervention times for teachers at risk of low resource generation.

---

## 1. Problem Definition

**AI Problem:**  
Predicting the optimal time for an intervention with a teacher at risk of low resource generation on the **SyncSenta** platform. The goal is to proactively support educators before disengagement occurs, thereby maximizing the value they derive from AI-powered tools.

### Objectives:
1. **Reduce Teacher Inactivity:** Increase the monthly active usage rate of the resource generation feature.
2. **Improve Resource Quality:** Identify teachers likely to produce low-rated or non-CBC-compliant resources, enabling pre-emptive guidance.
3. **Optimize Support Staff Allocation:** Prioritize human support efforts toward high-risk, high-impact teachers to maximize efficiency.

### Stakeholders:
- **Teachers:** Benefit from timely, personalized interventions that enhance their teaching experience and digital literacy.
- **SyncSenta Administrators / Product Team:** Responsible for user retention, product adoption, and demonstrating impact to partners and funders.

### Key Performance Indicator (KPI):
**Intervention Success Rate:**  
The percentage of predicted "at-risk" teachers who, after receiving a targeted intervention (e.g., in-app tutorial, peer mentorship invite, or direct support check-in), show measurable improvement within one week:
- Generate **≥5 new educational resources**, **or**
- Achieve a **≥15% increase in average resource quality score** (as rated by peers or automated CBC alignment checks).

---

## 2. Data Collection & Preprocessing

### Data Sources:
1. **Platform Interaction Logs:**  
   - Time spent on the resource creation interface  
   - Frequency of logins and session duration  
   - Number of resources generated per week  
   - Usage of AI-assist features (e.g., “CBC Alignment Check” tool)

2. **User Metadata and Behavioral History:**  
   - School type (public/private), county, teaching level  
   - Historical performance data (average resource ratings, compliance scores)  
   - Past engagement with support (ticket history, training completion)

### Potential Bias:
**Digital Access/Infrastructure Bias:**  
Teachers in rural or under-resourced schools may generate fewer resources not due to lack of skill, but because of poor internet connectivity or limited device access. A model trained without accounting for this could unfairly label them as "low performers," leading to misdirected interventions and widening equity gaps.

### Preprocessing Steps:
1. **Feature Scaling (Normalization):**  
   Apply Min-Max scaling to numerical features (e.g., `time_spent`, `resources_generated`) to ensure uniform contribution during model training.

2. **Handling Missing Data (Imputation):**  
   For new users lacking historical data (e.g., missing past quality scores), impute using median values or create a binary flag (e.g., `is_new_user = True`) to preserve context without data loss.

3. **Feature Encoding (One-Hot):**  
   Convert categorical variables like `county` and `school_type` into binary vectors to make them compatible with machine learning algorithms.

---

## 3. Model Development

### Chosen Model & Justification:
**Model:** **XGBoost (eXtreme Gradient Boosting)**  
**Justification:**  
XGBoost excels with structured tabular data and handles mixed feature types effectively. It offers high accuracy, robustness to outliers, and built-in regularization to prevent overfitting—critical when making decisions about human intervention. Its performance in real-world classification tasks makes it ideal for predicting teacher risk levels in a production environment.

### Data Splitting Strategy:
Use a **time-based split** to simulate real deployment conditions:
- **Training Set (70%):** All data up to Month $N-2$  
- **Validation Set (15%):** Data from Month $N-1$ (for hyperparameter tuning)  
- **Test Set (15%):** Most recent month ($N$) — evaluates generalization to future behavior

This avoids leakage and ensures the model can predict truly unseen future outcomes.

### Hyperparameters to Tune:
1. **`n_estimators` (Number of Trees):**  
   Controls model complexity and predictive power. Too many trees cause overfitting; too few lead to underfitting. Tuning balances accuracy and inference speed.

2. **`max_depth` (Maximum Tree Depth):**  
   Limits the depth of individual decision trees. Shallow trees generalize better; deeper ones capture complex patterns but risk memorizing noise. Critical for maintaining model reliability across diverse user groups.

---

## 4. Evaluation & Deployment

### Evaluation Metrics:
1. **Precision:**  
   $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$  
   Measures how often the model is correct when flagging a teacher as high-risk. High precision ensures efficient use of limited human support staff.

2. **Recall:**  
   $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$  
   Measures how many actual at-risk teachers were caught by the system. High recall prevents missed opportunities for impactful support.

> *Trade-off Note:* Depending on operational costs, we may prioritize precision (to avoid wasting staff time) or recall (to minimize disengagement). This balance will be determined collaboratively with stakeholders.

### Concept Drift:
**Definition:**  
Concept drift occurs when the relationship between input features and the target variable changes over time. For example, if Kenya introduces a revised CBC curriculum, what constitutes a "high-quality" resource shifts, invalidating historical labels.

### Monitoring Post-Deployment:
- **Performance Dashboard:** Track precision and recall weekly. Trigger alerts for >5% drops.
- **Data Drift Detection:** Monitor statistical shifts in key inputs (e.g., mean login frequency, distribution of school types).
- **Automated Retraining Pipeline:** Schedule retraining cycles (e.g., monthly) or trigger based on drift detection to keep the model current.

### Technical Challenge During Deployment:
**Real-Time Low-Latency Inference in Serverless Environment:**  
Deploying XGBoost in a serverless function (e.g., AWS Lambda) requires optimization to deliver predictions instantly upon user login. Challenges include cold starts, memory limits, and computational load.

**Solution Approach:**
- Serialize model using lightweight formats (e.g., ONNX or joblib with compression).
- Use provisioned concurrency to reduce cold start delays.
- Cache frequent predictions where appropriate.
- Aligns with focus on scalable, cost-effective serverless architecture.

---
