import numpy as np, logging
from sklearn.metrics import average_precision_score

def score(y_true, y_pred_proba):
    ''' 
    Return the area under the Precision-Recall curve.  
    Args:
        - y_true (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the ground truth observations (second column).
        - y_pred_proba (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the predicted probabilities estimates for the minority class (second column).
    Returns:
        float
    '''   
    y_true_sorted = y_true.sort_values(by='ID').reset_index(drop=True)
    y_pred_proba_sorted = y_pred_proba.sort_values(by='ID').reset_index(drop=True)
    score = average_precision_score(np.ravel(y_true_sorted.iloc[:, 1]), np.ravel(y_pred_proba_sorted.iloc[:, 1]))
    logging.info(f'score={score}')
    print(f'score={score}')
    return score