#! /bin/sh
python compressor_audit.py -data audit_tmp.npy -gpu 0 -data_params audit_tmp.params.json -model audit_tmp.hdf5 -model_name LSTM_multi_bn -output audit_tmp.compressed -batch_size 1000
wait
rm audit_tmp.npy
gzip audit_tmp.params.json
gzip audit_tmp.hdf5 
