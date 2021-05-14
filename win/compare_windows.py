f1=open('../../log_windows_1.json','r')
f2=open('win_decompress','r')
data1=f1.read()
data2=f2.read()
data1=data1.split('}\n{')[1:-1]
data2=data2.split('}\n{')
data2[0]=data2[0][1:]
data2[-1]=data2[-1][:-2]
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
        if 'PID' in json_obj.keys():
            pid_[json_obj['PID']+'*'+json_obj['Parent PID']+'*'+json_obj['Process Name']].append(data[i])
    for i in range(len(data)):
        json_obj=data[i]
        json_obj=json.loads(json_obj)
        if 'PID' in json_obj.keys():
            if pid_[json_obj['PID']+'*'+json_obj['Parent PID']+'*'+json_obj['Process Name']] != -1:
                for index in pid_[json_obj['PID']+'*'+json_obj['Parent PID']+'*'+json_obj['Process Name']]:
                    data_pro.append(index[1:-1])
                pid_[json_obj['PID']+'*'+json_obj['Parent PID']+'*'+json_obj['Process Name']]=-1
        else:
            data_pro.append(data[i][1:-1])
    return data_pro
data1=exec(data1)
if len(data1)!=len(data2):
    print(len(data1))
    print(len(data2))
    print('len 2 error')
    exit()
for i in range(len(data1)):
    if i%100000==0:
        print(i)
    data1[i]=data1[i].replace('\\\\u','\\u')
    data2[i]=data2[i].replace('\\\\u','\\u')
    if data1[i] !=data2[i]:
        print(i)
        print(data1[i])
        print(data2[i])
        print('error3')
        exit()
print('success')
