# ELISE-2021
This repository is source code for ELISE.

## Datasets
We provide part of our datasets in . Because log files contain a lot of sensitive information, we removed them for privacy concerns.

## Deployment
We use similiar enviroment as DeepZip and DeepZip provide a docker enviroment (We copy the makefile file and dockerfile from DeepZip).

### Install docker

- For bash enviroment

```shell
cd docker
make bash
```

- For other enviroments, please refer to DeepZip.
- For reuse of contaner.
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
nohup python -u trainer_elise.py -gpu 0 -file_type ftp -param ./ftp/ftp_tmp.params.json -d ./ftp/ftp_tmp.npy -name ./ftp/ftp_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv >log_ftp_1.file &

nohup python -u trainer_elise.py -gpu 0 -param ./bsd/darpa_tmp.params.json -d ./bsd/darpa_tmp.npy -name ./bsd/darpa_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv  >darpa_1.file &

nohup python -u trainer_elise.py -file_type httpd -gpu 0 -epoch 4 -d ./http/httpd_tmp.npy -param ./http/httpd_tmp.params.json -name ./http/httpd_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

nohup python -u trainer_elise.py -gpu 0 -file_type windows -param ./win/win_tmp.params.json -d ./win/win_tmp.npy -name ./win/win_tmp.hdf5 -model_name LSTM_multi_bn -batchsize 4096 -log_file LSTM.log.csv &

#compression
nohup python -u compressor.py -data ftp_tmp.npy -gpu 0 -data_params ftp_tmp.params.json -model ftp_tmp.hdf5 -model_name LSTM_multi_bn -output ftp_tmp.compressed -batch_size 1000 &

#decompression
nohup python -u decompressor_ftp.py -output ftp_tmp -gpu 0 -model ftp_tmp.hdf5 -model_name LSTM_multi_bn -input_file_prefix ftp_tmp.compressed -batch_size 1000 &
```


Note here, the key patterns can be automaticly found when parsing the log into json format. For illustration purpose, we provide the extraction python file for our datasets to avoid users use parsing tools such as logstash which may increase difficulty. We also provide the processed key patterns files so users do not need to run extraction files.

### Some important settings
For preprocessing 4, the match of substrings is complicated. For the efficiency, we implement it with a simpler way by directly obtaining frequent long strings.

Also, we provide several compare files to help user compare whether the content before compression and after compression is consistent. Because of the preprocessing 2 sorts the log entries originaly sort the log entires into different sessions, to keep exactly the same sequence of log entries, users can sort the log entries with their sequence ids/timestamps or create additional indexes during preprocessing.

### Other resources

The implementation of DeepZip can be found at https://github.com/mohit1997/DeepZip. We also provide a more memory effecient training file (trainer.py).
