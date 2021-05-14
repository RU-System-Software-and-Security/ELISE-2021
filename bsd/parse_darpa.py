import operator
from collections import defaultdict
import pickle
import sys
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-param_file', action='store', dest='param_file',
                    help='param file file')
parser.add_argument('-input', action='store', dest='input_file_path',
                    help='input file path')
parser.add_argument('-output', action='store',dest='output_file_path',
                    help='output file path')
parser.add_argument('-number', action='store',dest='number',help='number')
args = parser.parse_args()
conflict_dict={}
with open(args.input_file_path, 'r') as fp:
    data = fp.read()
data=str(data)
from collections import Counter
import re
import time

def get_dict_allkeys_values(dict_a,values,types,types1):
        for x in range(len(dict_a)):
            temp_key = dict_a[x][0]
            temp_value = dict_a[x][1]
            if temp_key==["datum","com.bbn.tc.schema.avro.cdm20.Event","type"]:
                if temp_value not in types:
                    types.append(temp_value)
            elif temp_key==['type']:
                if temp_value not in types1:
                    types1.append(temp_value)
            else:
                if temp_value.isnumeric():
                    if str(temp_key) not in conflict_dict.keys():
                        conflict_dict[str(temp_key)]=[]
                    conflict_dict[str(temp_key)].append(temp_value)
                if len(str(temp_value))>2:
                    values.append(str(temp_value)+'**'+str(temp_key))
        return values


def transfer(json_obj,dict_key,strr=[]):
    tmp_keys=list(json_obj.keys())
    for i in tmp_keys:
        tmp_values=json_obj[i]
        if isinstance(tmp_values,dict)==False  or len(list(tmp_values.keys()))==0:
            strr_=strr[:]
            strr_.append(i)
#            print(strr_)
            dict_key.append([strr_,str(tmp_values)])
        else:
            strr_=strr[:]
            strr_.append(i)
            transfer(tmp_values,dict_key,strr=strr_)
    return dict_key
global mins
mins=[]
time_obj=data.split("\n")[1]
def get_time(json_obj):
    dict_key=[]
    json_obj=json.loads(json_obj)
    tmp_data_time=transfer(json_obj,dict_key)
    tmp_data_time=dict_key
    dict_key=[]
    for i in tmp_data_time:
        if i[0]==["datum","com.bbn.tc.schema.avro.cdm20.Event","timestampNanos"]:
            print(i[1])
            return int(i[1])

mins.append(get_time(time_obj))
import sys
def count(data):
    data_=data.split("\n")[1:-1]#[1200000:]
#    data_[0]=data_[0][1:]
    values=[]
    types=[]
    types1=[]
    for i in range(len(data_)):
        dict_key=[]
        print(data_[i])
        json_obj=json.loads(data_[i])
        tmp_data_seq=transfer(json_obj,dict_key)
        tmp_data_seq=dict_key
        get_dict_allkeys_values(tmp_data_seq,values,types,types1)
    return Counter(values),types,types1

global re_keys
global re_values
values,types,types1=count(data)
values=sorted(values.items(), key=lambda values:values[1]*len(values[0].split("**")[0]), reverse=True)
values=dict(values)
tmp_dict={}
tmp_keys=list(values.keys())
for i in range(len(tmp_keys)):
    tmp_keys[i]=tmp_keys[i].split('**')[0]
    if tmp_keys[i] not in tmp_dict.keys():
        tmp_dict[tmp_keys[i]]=[i]
    else:
        tmp_dict[tmp_keys[i]].append(i)
tmp_keys=list(values.keys())
tmp_keys2=list(tmp_dict.keys())   #value [1,2,3]
final_dict={}

for i in range(len(tmp_keys2)):
    index=tmp_keys2[i]
    count=0
    for j in tmp_dict[tmp_keys2[i]]:
        index=index+'**'+tmp_keys[j].split("**")[1]
        count=count+values[tmp_keys[j]]*len(tmp_keys[j].split("**")[0])
    final_dict[index]=count
final_dict=sorted(final_dict.items(), key=lambda final_dict:final_dict[1], reverse=True)
minsss=0
for i in range(len(final_dict)):
    if int(final_dict[i][1])<20000:
        break
    else:
        minsss=minsss+1

final_dict=final_dict[:minsss] 
re_values=[]
re_keys={}
for name in conflict_dict.keys():
    re_keys[name]=[]
for (i,c) in final_dict:
    tmpp=i.split("**")
    re_values.append(tmpp[0])
   # re_keys.append(tmpp[1:])
    for j in tmpp[1:]:
        conflict_value=[]
        if str(j) in conflict_dict.keys():
            conflict_value=conflict_dict[str(j)]
        if j not in re_keys.keys():
            re_keys[j]=[]
            while str(len(re_keys[j])) in conflict_value:
                re_keys[j].append('**-1**')
            re_keys[j].append(tmpp[0])
        else:
            while str(len(re_keys[j])) in conflict_value:
                re_keys[j].append('**-1**')
            re_keys[j].append(tmpp[0])

import pickle
key_template=[]
with open('hpack_key_darpa'+str(args.number)+'.pickle','rb') as f:
    key_template=pickle.load(f)
key_template_=[]
for i in key_template:
    key_template_.append(str(i))
key_template_dict={c:i for (i,c) in enumerate(key_template_)}
def process_pid(json_obj_key,char2id_dict,id2char_dict,data_processed):

    if json_obj_key not in char2id_dict:
        for tmpchar in json_obj_key:
            if tmpchar not in char2id_dict:
                end=len(char2id_dict)+3
                char2id_dict[tmpchar]=end
                id2char_dict[end]=tmpchar
                data_processed.append(end)
            else:
                data_processed.append(char2id_dict[tmpchar])
    else:
        data_processed.append(char2id_dict[json_obj_key])
    data_processed.append(0)

def getallkeys(obj):
    tmp_key_patterns=[]
    for i in obj:
        tmp_key_patterns.append(i[0])
    return tmp_key_patterns
def handle_block(data_,char2id_dict,id2char_dict,types,types1):
    data_processed=[]
    flag=0
    for i in range(len(data_)):
        json_obj=data_[i]
        json_obj=json.loads(json_obj)
        dict_key=[]
        json_obj=transfer(json_obj,dict_key)
        json_obj=dict_key
        tmplist=getallkeys(json_obj)
        pid_key_list=[['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId'],['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId','properties','map','ppid']]
        if flag==0:
            flag=1
            data_processed.append(3)
            for com_key in pid_key_list:
                temp_value=0
                for kk in json_obj:
                    if kk[0] == com_key:
                        temp_value=kk[1]
                        if temp_value in re_values:
                            temp_value=str(re_keys[str(com_key)].index(temp_value))
                        process_pid(temp_value,char2id_dict,id2char_dict,data_processed)
            data_processed.append(3)
        for k in str(key_template_dict[str(tmplist)]):
            if k not in char2id_dict:
                end=len(char2id_dict)+4
                char2id_dict[k]=end
                id2char_dict[end]=k
                data_processed.append(end)
            else:
                data_processed.append(char2id_dict[k])
        data_processed.append(0)
        for x in json_obj:
            temp_value=x[1]
            if x[0] in pid_key_list:
                continue
            if x[0]==["datum","com.bbn.tc.schema.avro.cdm20.Event","type"]:
                temp_value=str(types.index(temp_value))
            if x[0]==['type']:
                temp_value=str(types1.index(temp_value))
            if x[0]==["datum","com.bbn.tc.schema.avro.cdm20.Event","timestampNanos"]:
                temp_value=str(int(x[1])-mins[0])
            if temp_value in re_values and x[0] not in [["datum","com.bbn.tc.schema.avro.cdm20.Event","type"],['type'],["datum","com.bbn.tc.schema.avro.cdm20.Event","timestampNanos"]]:
                temp_value=str(re_keys[str(x[0])].index(temp_value))
            if temp_value not in char2id_dict:
                for tmpchar in temp_value:
                    if tmpchar not in char2id_dict:
                        end=len(char2id_dict)+4
                        char2id_dict[tmpchar]=end
                        id2char_dict[end]=tmpchar
                        data_processed.append(end)
                    else:
                        data_processed.append(char2id_dict[tmpchar])
            else:
                data_processed.append(char2id_dict[temp_value])
            data_processed.append(1)
        data_processed.append(2) 
    data_processed.append(3)   
    return data_processed

def handle_normal(json_obj,char2id_dict,id2char_dict,types,types1):
    dict_key=[]
    transfer(json_obj,dict_key)
    json_obj=dict_key
    data_processed=[]
    tmplist=getallkeys(json_obj)
    for k in str(key_template_dict[str(tmplist)]):
        if k not in char2id_dict:
            end=len(char2id_dict)+4
            char2id_dict[k]=end
            id2char_dict[end]=k
            data_processed.append(end)
        else:
            data_processed.append(char2id_dict[k])
    data_processed.append(0)        
    for x in json_obj:
        temp_value=x[1]
        if x[0]==["datum","com.bbn.tc.schema.avro.cdm20.Event","type"]:
            temp_value=str(types.index(temp_value))
        if x[0]==['type']:
            temp_value=str(types1.index(temp_value))
        if x[0]==["datum","com.bbn.tc.schema.avro.cdm20.Event","timestampNanos"]:
            temp_value=str(int(x[1])-mins[0])
        if temp_value in re_values and x[0] not in [["datum","com.bbn.tc.schema.avro.cdm20.Event","type"],['type'],["datum","com.bbn.tc.schema.avro.cdm20.Event","timestampNanos"]]:
            temp_value=str(re_keys[str(x[0])].index(temp_value))
        if temp_value not in char2id_dict:
            for tmpchar in temp_value:
                if tmpchar not in char2id_dict:
                    end=len(char2id_dict)+4
                    char2id_dict[tmpchar]=end
                    id2char_dict[end]=tmpchar
                    data_processed.append(end)
                else:
                    data_processed.append(char2id_dict[tmpchar])
        else:
            data_processed.append(char2id_dict[temp_value])
        data_processed.append(1)
    data_processed.append(2) 
    return data_processed


def exec(data):
    data_=data.split("\n")[1:-1]#[1200000:]
    pid_=defaultdict(list)
    for i in range(len(data_)):
        json_obj=data_[i]
        json_obj=json.loads(json_obj)
        dict_key=[]
        keyss=transfer(json_obj,dict_key)
        keyss=dict_key
        keyss=getallkeys(keyss)
        if ['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId','int'] in keyss:
            pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])].append(data_[i])
    return pid_


def addsdict(data):
    data_=data.split("\n")[1:-1]
    data_processed=[]
    char2id_dict={}
    id2char_dict={}
    pid_=exec(data)
    for i in range(len(data_)):
        json_obj=data_[i]
        json_obj=json.loads(json_obj)
        dict_key=[]
        keyss=transfer(json_obj,dict_key)
        keyss=dict_key
        keyss=getallkeys(keyss)
        if ['datum','com.bbn.tc.schema.avro.cdm20.Event','threadId','int'] in keyss:
            if pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])] != -1:
                data_processed.append(handle_block(pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])],char2id_dict,id2char_dict,types,types1))
                pid_[str(json_obj['datum']['com.bbn.tc.schema.avro.cdm20.Event']['threadId']['int'])]=-1
        else:
            data_processed.append(handle_normal(json_obj,char2id_dict,id2char_dict,types,types1))       

    return char2id_dict,id2char_dict,data_processed 


char2id_dict,id2char_dict,data_processed=addsdict(data)
params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict,'key_template_dict':key_template_dict,'mins':mins,'re_values_dict':[re_keys,re_values],'types':types,'types1':types1}

with open(args.param_file, 'w') as f:
    json.dump(params, f, indent=4)
out = [c for item in data_processed for c in item]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
np.save(args.output_file_path, integer_encoded)


