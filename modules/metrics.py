import pandas as pd
import numpy as np

from dtaidistance import dtw
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def RMSE(y_val,y_pred):
    return np.sqrt(mean_squared_error(y_val, y_pred))

def MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def SMAPE(y_true, y_pred):
    n = len(y_true)
    sum_error = 0.0
    for t in range(n):
        sum_error += abs(y_pred[t] - y_true[t]) / (abs(y_true[t]) + abs(y_pred[t]))
    return (100.0/n) * sum_error

def DTW(y_true, y_pred):
    """
     https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html

    If you are interested in comparing only the shape,
    and not the absolute differences and offset, you need to transform the data first.
    Z-normalization
    T = (T-np.mean(T))/np.std(T)
    
    Differencing
    Z-normalization has the disadvantage that constant baselines are not necessarily at the same level.
    The causes a small error but it accumulates over a long distance.
    To avoid this, use differencing (see the clustering K-means documentation for a visual example).
    
    series = dtaidistance.preprocessing.differencing(series, smooth=0.1)
    """
    # Just-in-time Z-normalization
    y_true = (y_true - np.mean(y_true)) / np.std(y_true)
    y_pred = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    
    return dtw.distance_fast(y_true, y_pred,use_pruning=True)

def Temporal_loss(y_true, y_pred, sigma=1):
    """
    Computes the temporal loss between two time series using a modified Time Distortion Index.
    
    Args:
        y_true (ndarray): Ground truth time series.
        y_pred (ndarray): Predicted time series.
        sigma (float): Width of the Gaussian kernel used for temporal weighting.
        
    Returns:
        float: Temporal loss value.
    """
    
    # Compute the time distortion index between y_true and y_pred
    tdi = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    
    # Compute the temporal weighting function
    t = np.arange(len(y_true))
    weights = np.exp(-(t - t[:, np.newaxis]) ** 2 / (2 * sigma ** 2))
    
    # Compute the weighted TDI
    weighted_tdi = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * np.abs(y_true))
    
    return weighted_tdi


def PRINT_SCORE(y_val, y_pred):
    
    if isinstance(y_val, np.ndarray):y_val=y_val.ravel()
    if isinstance(y_val, pd.DataFrame):y_val=y_val.values.ravel()
    if isinstance(y_val, pd.Series):y_val=y_val.values.ravel()
   
    if isinstance(y_pred, np.ndarray):y_pred=y_pred.ravel()
    if isinstance(y_pred, pd.DataFrame):y_pred=y_pred.values.ravel()
    if isinstance(y_pred, pd.Series):y_pred=y_pred.values.ravel()
        
    print("RMSE : ",RMSE(y_pred,y_val))
    print("MAPE : ",MAPE(y_pred,y_val))
    print("SMAPE : ",SMAPE(y_pred,y_val))
    print("DTW : ",DTW(np.double(y_pred),np.double(y_val)))
    

def SCORE_2(model_name, y_pred, y_val, window_size, steps_ahead, target_column, RESULT_3 ):
    
    if isinstance(y_val, np.ndarray):y_val=y_val.ravel()
    if isinstance(y_val, pd.DataFrame):y_val=y_val.values.ravel()
    if isinstance(y_val, pd.Series):y_val=y_val.values.ravel()
   
    if isinstance(y_pred, np.ndarray):y_pred=y_pred.ravel()
    if isinstance(y_pred, pd.DataFrame):y_pred=y_pred.values.ravel()
    
    temp = pd.DataFrame({"model_name": str(model_name),
                         "window_size" :[window_size],
                         "steps_ahead" :[steps_ahead],
                         "target_column":[target_column],
                    "RMSE": [RMSE(y_pred, y_val)],
                    "MAPE": [MAPE(y_pred, y_val)],
                    "SMAPE": [SMAPE(y_pred, y_val)],
                    "DTW": [DTW(np.double(y_pred), np.double(y_val))]})

    RESULT_3 = pd.concat([RESULT_3, temp])
    return RESULT_3

if __name__ == "__main__":
    
    RESULT  = pd.DataFrame(columns=["model_name","RMSE","MAPE","SMAPE","DTW","RdR_score"])
    RESULT_2= pd.DataFrame(columns=["model_name","RMSE","MAPE","SMAPE","DTW"])
    RESULT_3= pd.DataFrame(columns=["model_name","window_size","steps_ahead","target_column","RMSE","MAPE","SMAPE","DTW"])