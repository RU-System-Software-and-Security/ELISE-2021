# ELISE-2021
This repository is the source code for ELISE.

## Datasets
We provide part of our datasets in . Because log files contain a lot of sensitive information, we removed them for privacy concerns.

## Deployment
We use a similar environment as DeepZip and DeepZip provides a docker environment (We copy the makefile file and docker file from DeepZip).

### Install docker

- For bash environment

```shell
cd docker
make bash
```

- For other environments, please refer to the implementation of DeepZip.
- For reuse of container.
```shell
docker start containerID
docker exec -it containerID /bin/bash
```

### Example of compression.

- Audit log file.
```shell
cd audit
#preprocessing
python extract_audit.py -input audit_log_1
python parse_audit.py -input audit_log_1 -output audit_tmp.npy -param audit_tmp.params.json
#training
cd ../
python trainer_elise.py -gpu 0 -param audit/audit_tmp.params.json -d audit/audit_tmp.npy -name audit/audit_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv
#compression
cd audit
./compress_audit.sh
#decompression
./decompress_audit.sh
#check correctness
python compare_audit.py
```
- Different log files.
```shell
#preprocessing
nohup python -u parse_ftp.py -input ../../ftp_1 -param ftp_tmp.params.json -output ftp_tmp.npy &

nohup python -u parse_darpa.py -input ../../darpa1 -number 1 -param darpa_tmp.params.json -output darpa_tmp.npy &

#training 
nohup python -u trainer_elise.py -gpu 0 -file_type ftp -param ./ftp/ftp_tmp.params.json -d ./ftp/ftp_tmp.npy -name ./ftp/ftp_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

nohup python -u trainer_elise.py -gpu 0 -param ./bsd/darpa_tmp.params.json -d ./bsd/darpa_tmp.npy -name ./bsd/darpa_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

nohup python -u trainer_elise.py -file_type httpd -gpu 0 -epoch 4 -d ./http/httpd_tmp.npy -param ./http/httpd_tmp.params.json -name ./http/httpd_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

nohup python -u trainer_elise.py -gpu 0 -file_type windows -param ./win/win_tmp.params.json -d ./win/win_tmp.npy -name ./win/win_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

#compression
nohup python -u compressor.py -data ftp_tmp.npy -gpu 0 -data_params ftp_tmp.params.json -model ftp_tmp.hdf5 -model_name LSTM_multi_bn -output ftp_tmp.compressed -batch_size 1000 &

#decompression
nohup python -u decompressor_ftp.py -output ftp_tmp_decompress -gpu 0 -model ftp_tmp.hdf5 -model_name LSTM_multi_bn -input_file_prefix ftp_tmp.compressed -batch_size 1000 &
```

Note that key patterns can be automatically found when parsing logs into JSON format. To illustrate, we provide a python file to extract key patterns of our datasets to avoid users additionally using a parsing tool such as logstash. We also provide processed key pattern files so that users do not need to run the code to obtain key patterns again.

### Some important settings
For preprocessing 4, matching of substrings is complicated. Considering the efficiency, we implement a simpler way to get frequent/common long strings directly.

Also, we provide several comparison files to help users compare the contents before and after compression. Since preprocessing 2 sorts the log entries and divides them into different sessions, to keep the exact same sequence of log entries, users can either sort log entries with their sequence ids/timestamps or create additional indexes during preprocessing.

### Other resources

The implementation of DeepZip can be found at https://github.com/mohit1997/DeepZip. We also provide a more memory efficient training file (trainer.py).
