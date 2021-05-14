#! /bin/sh
gunzip audit_tmp.params.json
gunzip audit_tmp.hdf5
wait
python decompressor_audit.py -output audit_log_1_decompress -gpu 0 -model audit_tmp.hdf5 -model_name LSTM_multi_bn -input_file_prefix audit_tmp.compressed -batch_size 1000
