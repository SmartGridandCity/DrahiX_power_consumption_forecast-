# DrahiX_power_consumption_forecast

non-stationnary time series forecast benchmark using DrahiX power consumption, 

dataset : 1h granularity, 42 481 rows, from 2016 to 2021, 1 building divided in 7 zones (T1, ... T7) and the sum of all columns (TGBT)

Implementation of deep learning model with Tensorflow :
  CNN-LTSM
  Seq2Seq 
  TCN-LTSM
  Attention-BiLTSM
  Seq2Seq-Attention
  Multihead-Attention
  Time2Vec-BiLTSM
  Time2Vec-Transformer
  
  Benchmark from 1 to 24h, with 168 and 672 h of window data, on the 8 columns.
  
  
# DrahiX Power Consumption Forecast
This repository provides a non-stationary time series forecast benchmark using DrahiX power consumption dataset. The dataset has a 1-hour granularity and contains 42,481 rows from 2016 to 2021. The data is from a single building divided into 7 zones (T1 to T7) and the sum of all columns (TGBT).

## Implementation\n\n
The deep learning models for the forecast were implemented using TensorFlow. The following models were used:

- CNN-LTSM\n
- Seq2Seq\n
- TCN-LTSM\n
- Attention-BiLTSM\n
- Seq2Seq-Attention\n
- Multihead-Attention\n
- Time2Vec-BiLTSM\n
- Time2Vec-Transformer\n
The benchmark for the models ranges from 1 to 24 hours, with 168 and 672 hours of window data on the 8 columns.

## Dataset\n\n
The dataset can be found in the data directory. It is stored in a CSV format and can be accessed using pandas or any other tool for data analysis. The dataset has a total size of approximately 3.2 MB.

## Usage
To use the code, you need to have TensorFlow installed in your environment. The code is written in Python, and the main file for running the models is main.py. The models can be run using the following command:



Contributors
- [](https://github.com/Aurel456)\n
Contributor 2
Contributor 3
Acknowledgments
Dataset Source 
TensorFlow
