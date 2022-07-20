"""
========================================================================
© 2018 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
import tqdm
import pandas as pd
from scripts.utility import load_ml_model
from scripts.preprocess import split_and_parallelize

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

# Survival Analysis
def survival_worker(partition):
    result = []
    for ikn, group in tqdm.tqdm(partition.groupby('ikn')):
        earliest_date, latest_date = group['visit_date'].min(), group['visit_date'].max()
        status = 0
        if group['D_date'].notnull().any():
            latest_date = group['D_date'].max() # D_date should be all the same
            status = 1
        result.append((ikn, earliest_date, latest_date, status))
    return result

def compute_survival(event_dates):
    """
    Args: 
        event_dates (pd.DataFrame): columns are patient's ikn, death date, and chemo visit dates 
                                    (each chemo visit date constitute one row)
    """
    result = split_and_parallelize(event_dates, survival_worker, processes=4)
    surv_df = pd.DataFrame(result, columns=['ikn', 'start', 'end', 'status'])
    return surv_df

# Treatment Recommendation Based on Survival
def get_subgroup(df, name, regimens, cancer_codes, cancer_col='curr_topog_cd', palliatative_intent=True, first_treatment_course=True):
    df = df[df[cancer_col].isin(cancer_codes)]
    logging.info(f'{len(df)} sessions for {name}')
    
    if palliatative_intent:
        df = df[df['intent_of_systemic_treatment'] == 'P']
        logging.info(f'{len(df)} sessions for palliative intent')
        
    if first_treatment_course:
        df = df[df['chemo_cycle'] == 1]
        logging.info(f'{len(df)} sessions for first treatment course')
        
    print(f'\nRegimen counts: \n{df["regimen"].value_counts().head(n=20)}\n')
    
    df = df[df['regimen'].isin(regimens)]
    logging.info(f'{len(df)} sessions for regimens: {regimens}')
    
    return df

def get_recommendation(train, cancer_df, algorithm='XGB', target_type='30d Mortality'):
    """Get treatment/regimen recommendation based on minimizing risk of death
    """
    model = load_ml_model(train.output_path, algorithm)
    idx = train.target_types.index(target_type)
    
    # set up model input
    X = pd.concat([X for X, _ in train.data_splits.values()])
    X = X.loc[cancer_df.index]
    cols = X.columns
    regimen_cols = cols[cols.str.contains('regimen')]
    X[regimen_cols] = 0

    # get predictions for each regimen 
    results = {}
    for regimen in cancer_df['regimen'].unique():
        col = f'regimen_{regimen}'
        X[col] = 1 # assign all patients to this regimen
        pred = model.predict_proba(X)
        pred = pred[idx][:, 1]
        results[regimen] = pred
        X[col] = 0 # reset
    results = pd.DataFrame(results, index=X.index)
    
    # recommend regimens that resulted in lower predicted risk of death (or death within x days)
    recommended_regimens = results.idxmin(axis=1)
    return recommended_regimens

def evaluate_recommendation(event_dates, cancer_df, recommended_regimens):
    """
    Evaluate recommendation based on median survival time of groups that 
    aligned with recommendation and did not align with recommendation
    """
    mask = cancer_df['regimen'] == recommended_regimens
    follows_recommendation = cancer_df[mask]
    against_recommendation = cancer_df[~mask]
    
    # Mortality value counts
    mvc = pd.DataFrame([follows_recommendation['Mortality'].value_counts(), 
                        against_recommendation['Mortality'].value_counts()], index=['Follows Reco', 'Against Reco'])
    mvc.columns = ['Dead', 'Alive/Censored']
    print(f'{mvc}\n')
    
    for name, group in {'following': follows_recommendation, 'against': against_recommendation}.items():
        dates = event_dates.loc[group.index]
        survival_time = dates['D_date'] - dates['visit_date']
        logging.info(f'Median survival time for group {name} recommendation: {survival_time.median().days} days')
