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

Note here, the key patterns can be automaticly found when parsing the log into json format. For illustration purpose, we provide the extraction python file for our datasets to avoid users use parsing tools such as logstash which may increase difficulty. We also provide the processed key patterns files so users do not need to run extraction files.

### Other resources

The implementation of DeepZip can be found at https://github.com/mohit1997/DeepZip. We also provide a more memory effecient training file (trainer.py).
