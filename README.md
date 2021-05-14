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
```

Note here, the key patterns can be automaticly found when parsing the log into json format. For illustration purpose, we provide the extraction python file for our datasets to avoid users use parsing tools such as logstash which may increase difficulty. We also provide the processed key patterns files so users do not need to run extraction files.

### Some important settings
For preprocessing 4, the match of substrings is complicated. For the efficiency, we implement it with a simpler way by directly obtaining frequent long strings.

Also, we provide several compare files to help user compare whether the content before compression and after compression is consistent. Because of the preprocessing 2 sorts the log entries originaly sort the log entires into different sessions, to keep exactly the same sequence of log entries, users can sort the log entries with their sequence ids/timestamps or create additional indexes during preprocessing.

### Other resources

The implementation of DeepZip can be found at https://github.com/mohit1997/DeepZip. We also provide a more memory effecient training file (trainer.py).
