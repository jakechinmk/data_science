# Give Me Some Credit Competition

## Overview
This is a credit scoring task from [Kaggle Competition](https://www.kaggle.com/competitions/GiveMeSomeCredit). 
Objective here is to predict serious delinquency in 2 years (person experienced 90 days past due delinquency or worse)


## Folder Structure
```
├── readme.md
├── notebook
│   ├── 00_download_data.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_missing_value_importance.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_xgboost_model.ipynb
├── output
│   ├── 01-EDA-train-diff.html
│   ├── 01-EDA-train.html
│   ├── 01-EDA-valid.html
│   ├── 03-baseline_model-pipeline.pkl
│   ├── 03-baseline_model-test_df.csv
│   ├── train_df.csv
│   ├── train_df1.csv
│   ├── readme.md
```

## Data Summary
- Target Variable: serious_dlqin_2yrs (person experienced 90 days past due delinquency of worse)
- Features
  - Gauge financial literacy
    - age
- Gauge ability to repay debt
  - Lending Capability
    - number_of_open_credit_lines_and_loans (installment and credit cards)
    - number_real_estate_loans_or_lines (mortgage and real estate loans - secured loans kind of)
    - revolving_utilization_of_unsecured_lines (total balance on credit cards and personal lines of credit divided by sum of credit lines)
  - Family Commitment
    - number_of_dependents
    - monthly_income
    - debt_ratio
  - Historical performance of repaying debts
    - number_of_time_30_59_days_past_due_not_worse
    - number_of_time_60_89_days_past_due_not_worse
    - number_of_times_90_days_late


## Key Information Needed
- What are the loan products currently offered in this dataset? (BNPL, Unsecured Loan, Secured Loan, etc)
  - Loan Amount, Interest Rate, Tenure 
- Who are the customers? (New or Repeated)
- What are the current loss?
- Why the delinquency of 90 days? Is there any comparison done across different days?
- What are the screening criteria on credit policy?
- Any credit policy changed during this period?
- The dataset presented is it all data collected prior to the user journey?

The questions are revolving around few factors
- What are the external environment that is possibly impacting the model building process
- Deciding the data collected is relatable or not
- Evaluate the target variables
  - Eg. if there are customer 
    - who have delinquency in 89 days, vs 
    - those who have delinquency in 90 days up to date (due to their loan is still fresh) vs 
    - those who have pay later than 180 days but their serious delinquency is 0.
  - This might introduce some noise to the model and make it difficult to classify.  

## Insights
- Age can be a good predictor based on the difference plot between good payer and bad payer
- Revolving Utlization of Unsecured Line upon filtering on max 1 seems to be having more bad payers around
- Lower monthly income after filtering 99th percentile seems to be having more bad payers around

## Summary on Weird Insights
- Weird as if it's not matching the common logical process.
- Age is ranging from 0 to 109 (credit policy should have a minimum age and maximum age to gauge their risk appetite)
- Monthly Income is ranging from 0 to a very large value (suppose they wouldn't fell into the same loan product in general sense, and also if they have no income how could they take a loan)
- Number of dependent of 20 is kind of out of the picture (how big can the family be, it's probably a self declared data)
- Ratios like debt ratio or revolving utilization suppose to be in percentage which range from 0 to 1 but they are having values more than 1
- Number of loans or lines that is having does not make that much sense either. It's probably a duplicate entry issue
- Number of time XX - XX days past due or late kind of adding the number of times everyday when they exceeded.
- Number of time 60 - 89 days and Number of time 90 days late seems to be correlate with each other once the number become huge. It seems that the calculation of these columns might have some issues.

## Methodology
- Using a relatively traditional method to understand which variable is important and how the missing values / outlier impacting the model using [optbinning](http://gnpalencia.org/optbinning/index.html)
- This library will automatically bin the variables into different groups (instead of we doing the coarse classing method manually by looking at WoE or IV)
- Feature selection based on information value
- Baseline Model using a simple Logistic Regression Model

## Result (Baseline)
- The feature selected
  - number of times 90 days late
  - number of time 30-59 days past due
  - number of time 60-89 days past due
  - age
  - debt_ratio
  - monthly_income
  - number_real_estate_loans_or_lines
  - number_of_dependents
- Model Performance
  - It's not really suitable to use AUC over this part but since the competition is evaluating based on AUC. It's better to get a grasp on what is going on with AUC. 
  - AUC CV Mean: 0.82 and generally it's over 0.8. As expected the model is relatively stable (less variance more bias)
  - Based on the confusion matrix, we might want to know what are the
    - Cost for losing the entire loan (probability * loan amount)
    - Cost for losing the profit (loan amount * interest rate)
- Bucket Evaluation
  - Training Data suggested the threshold to be 0.45, which we will have our maximum KS (50.88%)
  - Based on threshold = 0.45, 
    - We can cover 70-80% of the training dataset, and only having 23% to 32% of the bad payer of total bad payer in the population.
    - We also can cover 70-80% of the test dataset, and only having 21% to 30% of the bad payer. The model is considerably stable.
- Strategy Suggested
  - Lowest Risk (bin 1-3) - higher loan amount, lower interest rate, longer tenure
  - Moderate Risk (bin 4-8) - original loan amount, original interest rate, original tenure
  - High Risk (bin 9) - lower loan amount, higher interest rate, shortere tenure
  - Rejected (bin 10) - since it fell into the discrimination threshold (0.76)

## Future Work
- Ensure data quality (with check script)
- Train a different model
- Use different feature processing technique
  - Cap the outliers
  - Imputing the missing values
    - median for monthly income and number of dependent
    - impute by age group generate the monthly income median
  - Transformation of numerical continuous variable (not recommended if we are not using linear model)
- Geenerate features
  - The remaining balance after deducting all the commitments
  - Income per dependents
- Try on other model methods
  - XGboost (WIP)
  - Ensemble
- Model Evaluation
  - Focus on recall metric (since we would not want to have more bad payers as the loan amount loss is higher than the profit gained)



