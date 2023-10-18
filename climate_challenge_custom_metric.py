"""
Metric for the regional forecast climate challenge
"""
import numpy as np
import pandas as pd
import sys

def climate_metric_function(dataframe_y_true, dataframe_y_pred):
    
    if dataframe_y_true.equals(dataframe_y_pred):
        return 999999    
    
    else:
        nline,_=dataframe_y_true.shape   
        # makes the metric function symmetric by automatically identifying y_true as
        # the dataframe with an empyt VARIANCE column
        if not all(dataframe_y_true.VARIANCE.isna()):
            dataframe_y_true, dataframe_y_pred = dataframe_y_pred, dataframe_y_true
        if not all(dataframe_y_true.VARIANCE.isna()):
            return 10e-10

        ndata=nline//(192)
        df_pred_mean=np.zeros([ndata,192])
        df_pred_variance=np.zeros([ndata,192])
        df_true=np.zeros([ndata,192])

        for _ in range(ndata):
            df_pred_mean[:,:]=np.array(dataframe_y_pred['MEAN']).reshape(ndata,192)
            df_pred_variance[:,:]=np.array(dataframe_y_pred['VARIANCE']).reshape(ndata,192)
            df_true[:,:]=np.array(dataframe_y_true['MEAN']).reshape(ndata,192)

        chi2 = (df_pred_mean-df_true)**2
        
        R2 = np.sum(chi2)/np.sum((df_true)**2)
        RELIABILITY = np.sqrt(np.mean(chi2/(df_pred_variance+10-9)))

        score=-(np.log(R2)+abs(np.log(RELIABILITY)))
        
        return score