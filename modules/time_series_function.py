import numpy as np
import pandas as pd
import pywt

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import BSpline
from tsmoothie.smoother import ExponentialSmoother, KalmanSmoother

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

from typing import Tuple, List


def create_features(df,Fourier_terms=True, B_spline_features=True):
    """
    Create time series features based on time series index, also can add :
    Fourier_terms
    B_spline_features
    """
    df = df.copy()

    # Extract time-based features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    
    # Extract Fourier terms
    
    if Fourier_terms:
        """
        Fourier terms to capture seasonality patterns in the data.
        The Fourier terms are sinusoidal functions of the day/hour of year,
        with different frequencies (2, 4, 6, 8 cycles per year) and transformations
        """
        dayofyear = df.index.dayofyear
        hour = df.index.hour
        freq24 = 2 * np.pi / 24
        freq365 = 2 * np.pi / 365

        df['sin24_2'] = np.sin(freq24 * hour * 2)
        df['sin24_4'] = np.sin(freq24 * hour * 4)
        df['cos24_2'] = np.cos(freq24 * hour * 2)
        df['cos24_4'] = np.cos(freq24 * hour * 4)
        df['tanh24_2'] = np.tanh(freq24 * hour * 2)
        df['tanh24_4'] = np.tanh(freq24 * hour * 4)
        df['sin24_6'] = np.sin(freq24 * hour * 6)
        df['cos24_6'] = np.cos(freq24 * hour * 6)
        df['tanh24_6'] = np.tanh(freq24 * hour * 6)

        df['sin365_2'] = np.sin(freq365 * dayofyear * 2)
        df['cos365_2'] = np.cos(freq365 * dayofyear * 2)
        df['tanh365_2'] = np.tanh(freq365 * dayofyear * 2)
        df['sin365_4'] = np.sin(freq365 * dayofyear * 4)
        df['cos365_4'] = np.cos(freq365 * dayofyear * 4)
        df['tanh365_4'] = np.tanh(freq365 * dayofyear * 4)
        df['sin365_6'] = np.sin(freq365 * dayofyear * 6)
        df['cos365_6'] = np.cos(freq365 * dayofyear * 6)
        df['tanh365_6'] = np.tanh(freq365 * dayofyear * 6)

        # Fit scaler only once and transform Fourier terms
        fourier_cols = ['sin24_2', 'sin24_4', 'cos24_2', 'cos24_4',
                         'tanh24_2', 'tanh24_4',"sin24_6",
                        "cos24_6","tanh24_6","sin365_2","cos365_2",
                        "tanh365_2","sin365_4","cos365_4","tanh365_4",
                        "sin365_6","cos365_6","tanh365_6"]
    
        scaler = StandardScaler()
        df[fourier_cols] = scaler.fit_transform(df[fourier_cols])
        
#         for x in fourier_cols:
#             scaler = StandardScaler()
#             df[x] = scaler.fit_transform(df[x].values.reshape(-1,1))
        

    
    if B_spline_features:
        # Create B-spline features based on time series index
        """
        B-splines: B-splines are a family of piecewise polynomial functions that
        are commonly used in computer graphics, numerical analysis, and computational geometry
        to represent curves and surfaces. B-splines can be used to represent periodic signals by fitting a linear combination of
        basis functions to the signal. The basis functions are defined by a set of control points and a degree parameter,
        and they are non-zero only on a small interval of the signal. The B-spline representation can be more flexible
        than the Fourier or polynomial representations because it can capture both sharp transitions and smooth variations in 
        the signal.

        The function first creates a sequence of knot points t evenly spaced along the time series index.
        It then creates a B-spline object spl with n_knots basis functions of degree degree using the knot points as input.
        Finally, it evaluates the B-spline object at each time index x to obtain the B-spline basis functions
        and adds them as columns to the DataFrame with names 'b_spline_0', 'b_spline_1', etc.

        """
        x = df.index.values.astype(float)
        degree=5
        n_knots=8

        t = np.linspace(x.min(), x.max(), n_knots + degree - 1)
        spl = BSpline(t, np.eye(n_knots), k=degree)
        for i in range(n_knots-4):
            col_name = f'b_spline_{i}'
            df[col_name] = spl(x)[:, i]
    
    df.dropna(inplace=True)
    
    return df

def check_missing_dates(df,freq='H'):
    """
    freq can have '6H' or 'D' or any other form.......
    """
    # Get the start and end dates of the dataset
    start_date = df.index[0]
    end_date = df.index[-1]
    
    # Create a date range from the start to end date
    all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Compare the date range to the index of the dataframe
    missing_dates = set(all_dates) - set(df.index)
    
    # Print the missing dates, if any
    if missing_dates:
        print("Missing dates:")
        for date in missing_dates:
            print(date)
    else:
        print("No missing dates.")


##########    DENOISING    #########################################


def denoise_signal(signal, wavelet='db4', level=4):
    """
    Denoises a 1D input signal using Discrete Wavelet Transform (DWT) and soft thresholding.

    Parameters:
    signal : array_like
        Input 1D signal to be denoised.
    wavelet : str, optional
        Wavelet to be used in the DWT. Default is 'db4'.
    level : int, optional
        Level of decomposition in the DWT. Default is 4.

    Returns:
    denoised_signal : array_like
        Denoised 1D signal.

    Description:
    This function decomposes the input signal into approximation and detail coefficients using DWT.
    Soft thresholding is applied to the coefficients to remove noise.
    The denoised signal is reconstructed and returned.
    """
    # Decompose signal using wavelet transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Set threshold level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding
    for i in range(len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, 'soft')

    # Reconstruct signal using denoised coefficients
    denoised_signal = pywt.waverec(coeffs, wavelet)

    # Truncate denoised_signal to match df.index length
    # denoised_signal = denoised_signal[:len(df.index)]
    denoised_signal = denoised_signal[:len(signal)]

    

    return denoised_signal

def denoise_dataframe(df, columns, wavelet='db4', level=4):
    """
    Denoises specified columns in a DataFrame using Discrete Wavelet Transform (DWT) and soft thresholding.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing the signals to be denoised.
    columns : list
        List of column names in the DataFrame to be denoised.
    wavelet : str, optional
        Wavelet to be used in the DWT. Default is 'db4'.
    level : int, optional
        Level of decomposition in the DWT. Default is 4.

    Returns:
    df : pandas.DataFrame
        DataFrame with denoised signals as new columns (original_column_name_denoised).

    Description:
    This function denoises specified columns in the input DataFrame using DWT and soft thresholding.
    New columns are added to the DataFrame with '_denoised' appended to the original column names.
    """
    
    df = df.copy()
    for col in columns:
        df[col+'_denoised'] = denoise_signal(df[col].values, wavelet=wavelet, level=level)
    return df

# columns_to_denoise = df.columns
# df_wavelet_denoised = denoise_dataframe(df, columns=columns_to_denoise)
# # Plot original and denoised signals
# fig, axes = plt.subplots(nrows=len(columns_to_denoise), ncols=1, figsize=(40, 5*len(columns_to_denoise)))
# for i, col in enumerate(columns_to_denoise):
#     axes[i].plot(df_wavelet_denoised.index.values, df_wavelet_denoised[col], label='Original '+col)
#     axes[i].plot(df_wavelet_denoised.index.values, df_wavelet_denoised[col+'_denoised'], label='Denoised '+col)
#     axes[i].legend()
# plt.show()



def smooth_data_ExponentialSmoother(df, window_len=24*7, alpha=0.7):
    """
    Smooths the input DataFrame using ExponentialSmoother.
    Notes : you are loosing the first column of your DataFrame.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing the data to be smoothed.
    window_len : int, optional
        Length of the smoothing window. Default is 24*7.
    alpha : float, optional
        Smoothing factor. Default is 0.7.

    Returns:
    smooth_data : pandas.DataFrame
        Smoothed DataFrame.
    
    """
    smoother = ExponentialSmoother(window_len=window_len, alpha=alpha)
    smoother.smooth(df.T)
    smooth_data = pd.DataFrame(smoother.smooth_data.T, columns=df.columns, index=df.index[window_len:])
    return smooth_data


def smooth_data_Kalman(df, component='level_longseason', component_noise={'level':0.5, 'longseason':0.5}, n_longseasons=365):
    """
    Smooths the input DataFrame using KalmanSmoother.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing the data to be smoothed.
    component : str, optional
        Type of component to smooth. Default is 'level_longseason'.
    component_noise : dict, optional
        Noise levels for each component. Default is {'level':0.5, 'longseason':0.5}.
    n_longseasons : int, optional
        Number of long seasons. Default is 365.

    Returns:
    smooth_data : pandas.DataFrame
        Smoothed DataFrame.
    """
    smoother = KalmanSmoother(component=component, component_noise=component_noise, n_longseasons=n_longseasons)
    smoother.smooth(df.T)
    smooth_data = pd.DataFrame(smoother.smooth_data.T, columns=df.columns, index=df.index)
    return smooth_data


########   ROLLING STATISTICS


def Rolling_statistics(data, window, target):
    """
    Calculate rolling statistics including mean, standard deviation, skewness, and kurtosis for a specified target column
    in a pandas DataFrame.

    Parameters:
    data (pd.DataFrame): Input data as a pandas DataFrame.
    window (int): Size of the rolling window for computations.
    target (str): Name of the target column in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame with computed rolling statistics.

    Raises:
    ValueError: If input data is not a pandas DataFrame, window is not a positive integer, 
                target column does not exist, or window size is larger than the DataFrame length.
    """
    # Parameter validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data parameter must be a pandas DataFrame")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window parameter must be a positive integer")
    if target not in data.columns:
        raise ValueError("target parameter must be a valid column in the data DataFrame")

    # Error handling
    if len(data) < window:
        raise ValueError("window size cannot be larger than the length of the DataFrame")

    # Compute rolling statistics
    data['rolling_24_mean'] = data[target].rolling(window=24).mean()
    data['rolling_24_std']  = data[target].rolling(window=24).std()
    

    # Compute rolling skewness and kurtosis over a window of size 24
    rolling_24_skew = data[target].rolling(window=24, closed = 'both').apply(lambda x: pd.Series(x).skew())
    rolling_24_kurt = data[target].rolling(window=24, closed = 'both').apply(lambda x: pd.Series(x).kurt())
    data['rolling_24_skew'] = rolling_24_skew
    data['rolling_24_kurt'] = rolling_24_kurt
    
    if window >= 7*24:
        data['rolling_7day_mean'] = data[target].rolling(window=7*24).mean()
        data['rolling_7day_std'] = data[target].rolling(window=7*24).std()

        if window >= 7*24*4:
            data['rolling_4week_mean'] = data[target].rolling(window=24*7*4).mean()
            data['rolling_4week_std'] = data[target].rolling(window=24*7*4).std()
            
            # # Compute rolling skewness and kurtosis over a window of size 24*7*4
            # rolling_skewness = []
            # rolling_kurtosis = []

            # for i in range(window, len(data)+1):
            #     vals = data.iloc[i-window:i, :][target].values[[0, 24, 48, 72]]
            #     rolling_skewness.append(pd.Series(vals).skew())
            #     rolling_kurtosis.append(pd.Series(vals).kurt())
            # data['rolling_4week_skew'] = pd.Series(rolling_skewness, index=data.index[window-1:])
            # data['rolling_4week_kurt'] = pd.Series(rolling_kurtosis, index=data.index[window-1:])

    data.dropna(inplace=True)

    return data


############### feature selection


def feature_selection_dataframe(data,target_column='TGBT' ,k_best_features = 10):
  data=data.copy()
  # Separate the target variable from the features
  y = data[target_column]
  # Define a list of columns to drop
  columns_to_drop = ['TGBT','T1','T2','T3','T4','T5','T6','T7']

  X = data.copy()
  # Check if each column exists in the DataFrame, and drop it if it does
  for column in columns_to_drop:
      if column in data.columns:
          X = X.drop(column, axis=1)


  # Select the top 10 features based on the F-test
  selector = SelectKBest(score_func=f_regression, k=k_best_features)
  X_new = selector.fit_transform(X, y)

  # Get the names of the selected features
  selected_features = X.columns[selector.get_support()]
  print(f"selected_features : {selected_features}")

  df_new = pd.DataFrame(X_new, columns=selected_features, index= data.index)
  df_new[target_column] = y

  return df_new

##################    UNIVARIATE



def delete_every_nth_row(df, n):
    """
    This function takes in a pandas DataFrame and deletes every nth row from it. The value of n represents the size of the window or the number of forecasted values.
    It returns a new DataFrame with every nth row deleted.

    Parameters:
        df : pandas DataFrame
            The input DataFrame.
        n : int
            The frequency of rows to be deleted.

    Returns:
        pandas DataFrame
            A new DataFrame with every nth row removed.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [7, 8, 9, 10, 11, 12]})
        new_df = delete_every_nth_row(df, n=2)

        # new_df:
        #    A   B
        # 0  1   7
        # 2  3   9
        # 4  5  11
    """

    mask = pd.Series([True if i % n == 0 else False for i in range(len(df))])
    mask.index = df.index
    return df[mask]


def shiftting_data(data:pd.DataFrame,n_in:int,n_out:int,Target_column:str,dropna:bool=True)-> Tuple[pd.DataFrame,List[str]]:
    """
    This function takes a DataFrame and creates new columns by shifting the values of the Target_column by a specified number of positions.
    It takes inputs such as the number of time steps to shift the Target_column backwards and forwards, the name of the Target_column, 
    and a boolean value indicating whether or not to drop the rows with missing values. It returns the modified DataFrame and a list of the names of the new columns.

    Parameters:
        data : pandas DataFrame
            The input DataFrame.
        n_in : int
            The number of time steps to shift the Target_column backwards.
        n_out : int
            The number of time steps to shift the Target_column forwards.
        Target_column : str
            The name of the target column to shift.
        dropna : bool, optional
            A boolean value indicating whether or not to drop rows with missing values. Default is True.

    Returns:
        shifted_df : pandas DataFrame
            The modified DataFrame with new columns representing the shifted values of the Target_column.
        new_columns : list of str
            A list of the names of the new columns.

    Example:
        df = pd.DataFrame({'Date': ['2021-01-01', '2021-01-02', '2021-01-03'], 'Value': [10, 20, 30]})
        shifted_df, new_columns = shiftting_data(df, n_in=2, n_out=3, Target_column='Value', dropna=True)

        # shifted_df:
        #         Date  Value  Value -1h  Value -2h  Value +1h  Value +2h  Value +3h
        # 2 2021-01-03     30       20.0       10.0        NaN        NaN        NaN

        # new_columns:
        # ['Value', 'Value -1h', 'Value -2h', 'Value +1h', 'Value +2h', 'Value +3h']
    """
    
    df=data.copy()
    TARGET=str(Target_column)
    col_name=[TARGET]
    
    for k in range(1,n_in+1):
        col=f'{TARGET} -{k}h'
        df[col]=df[TARGET].shift(k)
        col_name.append(col)
        
    for k in range(1,n_out+1):
        col=f'{TARGET} +{k}h'
        df[col]=df[TARGET].shift(-k)
        # col_name.append(col)
    
    df.dropna(inplace=True)
    
    return df,col_name


### ??????????????????????????????????????????????????????????????????????????????? Univariate_X_Y a modifier
from scipy.stats import skew
from scipy.stats import kurtosis
def Univariate_X_Y(df,window_size=24,target=24, Target_column = 'TGBT'):
    
    if Target_column != 'AirTemp':
        df = df[['AirTemp', Target_column]].copy()

        # create shifted dataset
        dataT,colname=shiftting_data(df,n_in=window_size-1,n_out=target,Target_column=Target_column)

        # # search for the temperature ----> creates Nan values ...
        # if window_size >=24:
        #     dataT2,_=shiftting_data(df,n_in=24-1,n_out=0,Target_column='AirTemp')

        # else:
        #     dataT2,_=shiftting_data(df,n_in=window_size-1,n_out=0,Target_column='AirTemp')

        # search for the temperature
        dataT2,_=shiftting_data(df,n_in=window_size-1,n_out=0,Target_column='AirTemp')
        dataT2 = dataT2[:-target]

        # fuse
        dataT = pd.concat([dataT2, dataT], axis=1, join='outer')
        #  pandas remove duplicate columns
        dataT = dataT.loc[:,~dataT.columns.duplicated()].copy()

        # delete repeting row based on target
        dataT = delete_every_nth_row(dataT,target)
        dataT['Skew'] = dataT[colname].apply(lambda x: skew(x), axis=1)
        dataT['kurtosis'] = dataT[colname].apply(lambda x: kurtosis(x), axis=1)

    else :
        df = df[['AirTemp']].copy()

        # create shifted dataset
        dataT,colname=shiftting_data(df,n_in=window_size-1,n_out=target,Target_column=Target_column)
        # delete repeting row based on target
        dataT = delete_every_nth_row(dataT,target)

        dataT['Skew'] = dataT[colname].apply(lambda x: skew(x), axis=1)
        dataT['kurtosis'] = dataT[colname].apply(lambda x: kurtosis(x), axis=1)


    # check_missing_dates(dataT,f'{target}H')
    
    # split train and validation
    n = len(dataT)
    train_T = dataT[0:int(n*0.9)]
    val_T = dataT[int(n*0.9):]
    
    # FEATURES and TARGET
    FEATURES=[col for col in dataT.columns if '+' not in col]
    TARGET=[col for col in dataT.columns if '+' in col]
    
    X_train_T=train_T[FEATURES]
    y_train_T=train_T[TARGET]

    X_val_T=val_T[FEATURES]
    y_val_T=val_T[TARGET]
    
    return X_train_T, y_train_T , X_val_T , y_val_T
 
    
# X_train, y_train , X_val , y_val = Univariate_X_Y(df = df_Kalman_filtered,window_size=24,target=24, Target_column = 'TGBT')
# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)

##################    MULTIVARIATE


  