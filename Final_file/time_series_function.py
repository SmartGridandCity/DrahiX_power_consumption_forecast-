import numpy as np
import pandas as pd
import pywt

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import BSpline
from tsmoothie.smoother import ExponentialSmoother, KalmanSmoother


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
    denoised_signal = denoised_signal[:len(df.index)]

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