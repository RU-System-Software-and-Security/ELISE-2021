f1=open('../../mysql/mysql_1','r')
f2=open('sql_tmp_decompress','r')
data1=f1.read()
data2=f2.read()
data1=data1.split('}{')[1:-1]
data2=data2.split('}{')
data2[0]=data2[0][1:]
data2[-1]=data2[-1][:-1]
if len(data1)!=len(data2):
    print('length error')
    exit()
import json
from collections import defaultdict
def exec(data):
    data_pro=[]
    pid_=defaultdict(list)
    for i in range(len(data)):
        data[i]="{"+data[i]+"}"
        json_obj=data[i]
        json_obj=json.loads(json_obj)
        if 'mysql-type' in json_obj.keys():
            pid_[json_obj['mysql-type']].append(data[i])
    for i in range(len(data)):
        json_obj=data[i]
        json_obj=json.loads(json_obj)
        if 'mysql-type' in json_obj.keys():
            if pid_[json_obj['mysql-type']] != -1:
                for index in pid_[json_obj['mysql-type']]:
                    data_pro.append(index[1:-1])
                pid_[json_obj['mysql-type']]=-1
        else:
            data_pro.append(data[i][1:-1])
    return data_pro

data1=exec(data1)
if len(data1)!=len(data2):
    print('len 2 error')
    exit()
for i in range(len(data1)):
    if i % 100000==0:
        print(i)
    if data1[i] !=data2[i]:
        print(i)
        print(data1[i])
        print(data2[i])
        print('error3')
        exit()
print('success')
