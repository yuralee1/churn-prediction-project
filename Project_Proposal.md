# Project Proposal: Predicting Customer Churn in Subscription-Based Streaming Services

## 1. Business Problem or Opportunity

Subscription-based streaming services operate in a highly competitive market where customer acquisition costs are significantly higher than retention costs. When a subscriber cancels their service (churns), the business not only loses recurring revenue but must also spend additional resources to acquire a replacement customer. Currently, many streaming providers respond to churn reactively, offering retention incentives only after a customer has already initiated a cancellation, at which point it is often too late to change their decision.

The core business problem this project addresses is the inability to **identify subscribers who are at high risk of churning before they cancel**. Without a reliable, data-driven method for flagging at-risk customers early, retention teams cannot prioritize their outreach, personalize offers, or intervene at the right moment in the customer lifecycle.

## 2. Project Objectives

This project aims to:

1. **Identify the customer attributes and behaviors** most strongly associated with churn (e.g., contract type, tenure, monthly charges, payment method, service usage).
2. **Build and evaluate a predictive classification model** capable of estimating the probability that an individual subscriber will churn.
3. **Translate model findings into actionable, business-ready recommendations** that a retention team can use to reduce churn and improve customer lifetime value.

## 3. Business Significance

Reducing customer churn has a direct and measurable impact on profitability. Industry research consistently shows that retaining an existing customer costs far less than acquiring a new one, and even modest reductions in churn rate can produce outsized gains in long-term revenue due to the compounding value of subscription relationships.

By proactively identifying at-risk subscribers, the business can:

- Target retention offers (discounts, plan adjustments, personalized outreach) only at customers who are actually likely to leave, improving marketing efficiency and ROI.
- Reduce revenue loss from preventable cancellations.
- Gain a deeper understanding of *why* customers churn, informing broader strategic decisions around pricing, contract structures, and product/engagement features.
- Support a shift from reactive customer service to proactive, data-informed retention strategy.

This project is therefore directly relevant to marketing, customer success, and executive stakeholders who are accountable for subscriber growth and retention metrics.

## 4. Description of the Proposed Dataset

This project will use the **IBM Telco Customer Churn dataset**, a widely used and publicly available dataset that closely mirrors the structure of subscription-service customer data (it was originally built for telecom churn analysis but is structurally analogous to streaming subscription data, which commonly uses the same dataset as a proxy in academic and applied analytics work).

The dataset includes approximately 7,000 customer records with variables such as:

- **Demographics:** gender, senior citizen status, dependents, partner status
- **Subscription details:** contract type (month-to-month, one-year, two-year), tenure (months as a customer), payment method
- **Billing information:** monthly charges, total charges
- **Service engagement:** internet service type, streaming/service add-ons, technical support usage
- **Target variable:** churn label (whether the customer canceled their subscription)

This dataset is well-suited to the project because it contains a labeled outcome variable (churn), a mix of categorical and numeric predictors, and a realistic level of data quality issues (missing values, class imbalance) that will allow meaningful application of data preparation and modeling techniques within the scope of this course.

## 5. Data Acquisition Strategy

The dataset will be acquired directly from its public, freely available source (Kaggle: "IBM Telco Customer Churn"). Because it is distributed as a single downloadable CSV file, no API access, scraping, or credentialing is required, which keeps acquisition low-risk and fully reproducible.

The acquisition process will include:

1. Downloading the dataset file directly from the source.
2. Verifying file integrity (row/column counts, expected fields) upon import.
3. Documenting the source, access date, and license/usage terms for transparency and reproducibility.

If time and access permit, this dataset may later be supplemented with organization-specific or additional publicly available streaming-service data to strengthen the business framing; however, the Telco dataset alone is sufficient to complete the project within the course timeline.

## 6. Preliminary Data Preparation Plan

Data preparation will proceed in two main phases, consistent with the approved project workflow:

**Data Preparation (Stage 2)**
- Load the raw dataset into the analysis environment.
- Consolidate relevant variables into a single analytic table (demographics, subscription plan, tenure, charges, payment method, engagement, support interactions, churn label).
- Identify and remove duplicate records.
- Confirm data types are correctly assigned (numeric, categorical, boolean).

**Preprocessing (Stage 3)**
- Handle missing values (e.g., imputation or removal, depending on the variable and missingness pattern).
- Encode categorical variables (e.g., one-hot encoding for contract type and payment method).
- Address class imbalance between churned and retained customers (e.g., resampling techniques or class-weighting, to be determined during modeling).
- Scale or normalize numeric fields (e.g., monthly charges, tenure) where required by the chosen modeling technique.

This staged approach ensures the dataset is both analytically sound and appropriately structured before modeling begins.

## 7. Proposed Analytical or AI Techniques

This project will apply **supervised classification techniques** to predict customer churn, including:

- **Logistic Regression** — as a baseline, interpretable model for estimating churn probability.
- **Decision Trees** — to capture non-linear relationships and provide interpretable decision rules.
- **Random Forest** — to improve predictive performance through ensemble learning and to help identify feature importance.

Model performance will be evaluated using **accuracy, precision, recall, and confusion matrix analysis**, with particular attention paid to recall and precision balance given the class imbalance between churned and retained customers (correctly identifying churners matters more than raw accuracy in this context).

Exploratory Data Analysis (EDA), using summary statistics and visualizations, will precede modeling to surface initial patterns (e.g., churn rates by contract type, tenure, or monthly charges) and inform feature selection.

## 8. Expected Project Outcomes

Upon completion, this project is expected to deliver:

1. A cleaned, analysis-ready dataset documenting all preparation and preprocessing steps.
2. A trained and validated churn prediction model, with a clear comparison of at least two classification approaches.
3. Visualizations summarizing churn drivers and model performance (e.g., feature importance charts, ROC curves, confusion matrix heatmaps).
4. A set of actionable business recommendations for the retention team, such as prioritizing high-risk customers for targeted offers, reconsidering contract or pricing structures, and informing engagement/service strategy.
5. A final written report and presentation documenting the full analytical process, findings, limitations, and next steps for stakeholders.

## 9. Preliminary Implementation Timeline

| Week | Milestone | Key Activities |
|------|-----------|-----------------|
| 1–2 | Data Acquisition & Preparation | Acquire dataset, verify integrity, consolidate and clean data (Stages 1–2) |
| 3 | Preprocessing | Handle missing values, encode variables, address class imbalance, scale features (Stage 3) |
| 4 | Exploratory Data Analysis | Generate summary statistics and visualizations; identify initial churn patterns (Stage 4) |
| 5–6 | Model Development | Build and tune logistic regression, decision tree, and random forest models (Stage 5) |
| 7 | Model Evaluation & Visualization | Evaluate models using accuracy, precision, recall, confusion matrix; build performance visualizations (Stages 5–6) |
| 8 | Interpretation & Recommendations | Translate findings into business insights and retention recommendations (Stages 7–8) |
| 9 | Final Reporting | Compile final report and presentation; document methodology, results, and limitations (Stage 9) |

This timeline aligns with the course schedule and reflects a feasible, incremental path from raw data to a decision-ready deliverable for the retention team.
