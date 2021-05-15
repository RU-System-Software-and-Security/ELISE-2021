nohup python -u compressor.py -data sql_tmp.npy -gpu 0 -data_params sql_tmp.params.json -model sql_tmp.hdf5 -model_name LSTM_multi_bn -output sql_tmp.compressed -batch_size 1000 &

