import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_saisonal_decomp(data):
    # Perform seasonal decomposition
    result = seasonal_decompose(data)

    # Extract the components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Create a dataframe with the original data and the decomposition components
    df_seasonality=pd.DataFrame({'original': data, 'trend': trend, 'seasonality': seasonal, 'residual': residual})



    fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(30,20))

    axes[0].plot(df_seasonality['trend'][:500],'r',label='trend')
    axes[0].plot(df_seasonality['seasonality'][:500],label='seasonality')
    axes[0].plot(df_seasonality['residual'][:500],label='residual')


    axes[1].plot(df_seasonality['trend'][3100:3600],'r',label='trend')
    axes[1].plot(df_seasonality['seasonality'][3100:3600],label='seasonality')
    axes[1].plot(df_seasonality['residual'][3100:3600],label='residual')

    axes[2].plot(df_seasonality['trend'][4000:4500],'r',label='trend')
    axes[2].plot(df_seasonality['seasonality'][4000:4500],label='seasonality')
    axes[2].plot(df_seasonality['residual'][4000:4500],label='residual')

    axes[3].plot(df_seasonality['trend'][-500:],'r',label='trend')
    axes[3].plot(df_seasonality['seasonality'][-500:],label='seasonality')
    axes[3].plot(df_seasonality['residual'][-500:],label='residual')

    plt.legend()
    plt.show()
