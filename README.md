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
#preprocessing
python extract_audit.py -input audit_log_1
#training

#compression

#decompression
```

Note here, the key patterns can be automaticly found when using logstash. For illustration purpose, we provide the extraction file for our datasets to avoid users use logstash which may increase difficulty. We also provide the processed key patterns files so users do not need to run extraction files.

### Other resources

The implementation of DeepZip can be found at https://github.com/mohit1997/DeepZip. We also provide more memory effecient training file, compression file and decompression file (trainer.py, compressor.py and decompressor.py).
