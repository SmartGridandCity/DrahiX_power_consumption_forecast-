# DrahiX_power_consumption_forecast

non-stationnary time series forecast benchmark using DrahiX power consumption, 

dataset : 1h granularity, 42 481 rows, from 2016 to 2021, 1 building divided in 7 zones (T1, ... T7) and the sum of all columns (TGBT)

Implementation of deep learning model with Tensorflow :
  CNN-LTSM \n
  Seq2Seq \n
  TCN-LTSM
  Attention-BiLTSM
  Seq2Seq-Attention
  Multihead-Attention
  Time2Vec-BiLTSM
  Time2Vec-Transformer
  
  Benchmark from 1 to 24h, with 168 and 672 h of window data, on the 8 columns. 
