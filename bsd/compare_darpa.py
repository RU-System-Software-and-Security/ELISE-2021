f1=open('../darpa10','r')
f2=open('../data/decompressed/darpa_10','r')
data1=f1.read()
data2=f2.read()
data1=data1.split('\n')[1:-1]
#data1[0]=data1[0][1:]
data2=data2.split('\n')[:-1]
#data2[0]=data2[0][1:]
#data2[-1]=data2[-1][:-1]
#data1[0]=data1[0][1:]
print(len(data1))
print(len(data2))
if len(data1)!=len(data2):
    #print(data1[-1])
    #print(data2[-1])
    print('length error')
    exit()
import json
from collections import defaultdict
def transfer(json_obj,dict_key,strr=[]):
    tmp_keys=list(json_obj.keys())
    for i in tmp_keys:
        tmp_values=json_obj[i]
        if isinstance(tmp_values,dict)==False  or len(list(tmp_values.keys()))==0:
            strr_=strr[:]
            strr_.append(i)
            dict_key.append([strr_,str(tmp_values)])
        else:
            strr_=strr[:]
            strr_.append(i)
            transfer(tmp_values,dict_key,strr=strr_)
    return dict_key

def getallkeys(obj):
    tmp_key_patterns=[]
    for i in obj:
        tmp_key_patterns.append(i[0])
    return tmp_key_patterns

def exec(data):
    data_pro=[]
    pid_=defaultdict(list)
    for i in range(len(data)):
   #     data[i]="{"+data[i]+"}"
#        print(data[i])
        json_obj=data[i]
        json_obj=json.loads(json_obj)
        dict_key=[]
        keyss=transfer(json_obj,dict_key)
        keyss=dict_key
        dict_key=[]
        keyss=getallkeys(keyss)
        if ['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId','int'] in keyss:
            pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])].append(data[i])
    for i in range(len(data)):
     #   data[i]="{"+data[i]+"}"
        json_obj=data[i]
        json_obj=json.loads(json_obj)
        dict_key=[]
        keyss=transfer(json_obj,dict_key)
        keyss=dict_key
        dict_key=[]
        keyss=getallkeys(keyss)
        if ['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId','int'] in keyss:
            if pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])] != -1:
                for index in pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])]:
                    data_pro.append(index)
                pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])]=-1
        else:
            data_pro.append(data[i])
    return data_pro

data1=exec(data1)
#print(data1)
if len(data1)!=len(data2):
    print('len 2 error')
    exit()
for i in range(len(data1)):
    if i%100000==0:
        print(i)
    if data1[i] !=data2[i]:
        print(i)
        print(data1[i])
        print(data2[i])
        print('error3')
#        exit()
print('success')
