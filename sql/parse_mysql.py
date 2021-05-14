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
args = parser.parse_args()
#import json

conflict_dict={}
with open(args.input_file_path, 'rb') as fp:
    data = fp.read()
data=str(data)
from collections import Counter
import re
import time
def translate_time(time_):
    day=time_.split('T')
    day_=day[0].split('-')
    year=day_[0]
    month=day_[1]
    day_=day_[2]
    timeArray = time.strptime(str(str(year)+'-'+str(month)+'-'+str(day_)+' '+day[1]), "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    return timestamp
global mins
json_tmp=json.loads('{'+data.split('}{')[1]+'}')
print(json_tmp)

mins=[]
mins.append(translate_time(json_tmp['timestamp']))
mins.append(int(json_tmp['counter'].split('Z')[0]))
def get_dict_allkeys_values(dict_a,values):
        if isinstance(dict_a, dict): 
            for x in range(len(dict_a)):
                temp_key = list(dict_a.keys())[x]
                temp_value = dict_a[temp_key]
                if str(temp_value)=='None':
                    values.append('null')
                elif str(temp_value)=='False':
                    values.append('false')
                else:
                    if temp_key not in ['timestamp','counter']:
                        if temp_value.isnumeric():
                            if str(temp_key) not in conflict_dict.keys():
                                conflict_dict[str(temp_key)]=[]
                            conflict_dict[str(temp_key)].append(temp_value)
                        if len(str(temp_value))>2:
                            values.append(str(temp_value)+'**'+str(temp_key))
        return values
import sys
def count(data):
    data_=data.split("}{")[1:-1]#[1200000:]
    values=[]
    for i in range(len(data_)):
        json_obj="{"+data_[i]+"}"
        json_obj=json.loads(json_obj)
        get_dict_allkeys_values(json_obj,values)
    return Counter(values)

global re_keys
global re_values
values=count(data)
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
final_dict=final_dict[:minsss]  #40
re_values=[]
re_keys={}
for (i,c) in final_dict:
    tmpp=i.split("**")
    re_values.append(tmpp[0])
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
print(re_keys)
print(re_values)

import pickle
key_template=[]
with open('hpack_key_mysql.pickle','rb') as f:
    key_template=pickle.load(f)
key_template_dict={c:i for (i,c) in enumerate(key_template)}

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


def handle_block(data_,char2id_dict,id2char_dict):
    data_processed=[]
    flag=0
    for i in range(len(data_)):
        json_obj=data_[i]
        json_obj=json.loads(json_obj)
        pid_key_list=['mysql-type']
        tmplist=list(json_obj.keys())
        if flag==0:
            flag=1
            common_key=[val for val in tmplist if val in pid_key_list]
            data_processed.append(2)
            for com_key in common_key:
                temp_value=json_obj[com_key]
                if temp_value in re_values:
                    temp_value=str(re_keys[com_key].index(temp_value))
                process_pid(temp_value,char2id_dict,id2char_dict,data_processed)
            data_processed.append(2)
        temp_key=[val for val in tmplist if val not in common_key]
        for x in temp_key:
            temp_value = str(json_obj[x])
            if x=='counter':
                if len(temp_value.split('Z'))>2:
                    print(temp_value)
                temp_value=str(int(temp_value.split('Z')[0])-mins[1])+'Z'+temp_value.split('Z')[1]
            if x=='timestamp':
                tmp_time=translate_time(temp_value)
                temp_value=str(int(tmp_time)-mins[0])
            if temp_value in re_values and x not in ['timestamp','counter']:
                temp_value=str(re_keys[x].index(temp_value))
            if temp_value not in char2id_dict:
                for tmpchar in temp_value:
                    if tmpchar not in char2id_dict:
                        end=len(char2id_dict)+3
                        char2id_dict[tmpchar]=end
                        id2char_dict[end]=tmpchar
                        data_processed.append(end)
                    else:
                        data_processed.append(char2id_dict[tmpchar])
            else:
                data_processed.append(char2id_dict[temp_value])
            data_processed.append(0)
        data_processed.append(1) 
    data_processed.append(2)   
    return data_processed
def handle_normal(json_obj,char2id_dict,id2char_dict):
    data_processed=[]
    tmplist=list(json_obj.keys())
    #print(tmplist)
    tmplist.sort()
    for x in range(len(tmplist)):
        temp_key = tmplist[x]
        temp_value = str(json_obj[temp_key])
        if temp_key=='counter':
            if len(temp_value.split('Z'))>2:
                print(temp_value)
            temp_value=str(int(temp_value.split('Z')[0])-mins[1])+'Z'
        if temp_key=='timestamp':
            tmp_time=translate_time(temp_value)
            temp_value=str(int(tmp_time)-mins[0])
        if temp_value in re_values and temp_key not in ['timestamp','counter']:
            temp_value=str(re_keys[temp_key].index(temp_value))
        if temp_value not in char2id_dict:
            for tmpchar in temp_value:
                if tmpchar not in char2id_dict:
                    end=len(char2id_dict)+3
                    char2id_dict[tmpchar]=end
                    id2char_dict[end]=tmpchar
                    data_processed.append(end)
                else:
                    data_processed.append(char2id_dict[tmpchar])
        else:
            data_processed.append(char2id_dict[temp_value])
        data_processed.append(0)
    data_processed.append(1)
    return data_processed

def exec(data):
    data_=data.split("}{")[1:-1]#[1200000:]
    pid_=defaultdict(list)
    for i in range(len(data_)):
        data_[i]="{"+data_[i]+"}"
        json_obj=data_[i]
        json_obj=json.loads(json_obj)
        if 'mysql-type' in json_obj.keys():
            pid_[json_obj['mysql-type']].append(data_[i])
    return pid_


def addsdict(data):
    data_=data.split("}{")[1:-1]#[1200000:]
    data_processed=[]
    char2id_dict={}
    id2char_dict={}
    pid_=exec(data)
    for i in range(len(data_)):
        json_obj="{"+data_[i]+"}"
        json_obj=json.loads(json_obj)
        if 'mysql-type' in json_obj.keys():
            if pid_[json_obj['mysql-type']] != -1:
                data_processed.append(handle_block(pid_[json_obj['mysql-type']],char2id_dict,id2char_dict))
                pid_[json_obj['mysql-type']]=-1
        else:
            data_processed.append(handle_normal(json_obj,char2id_dict,id2char_dict))       

    return char2id_dict,id2char_dict,data_processed 


char2id_dict,id2char_dict,data_processed=addsdict(data)
params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict,'key_template_dict':key_template_dict,'mins':mins,'re_values_dict':[re_keys,re_values]}

with open(args.param_file, 'w') as f:
    json.dump(params, f, indent=4)
out = [c for item in data_processed for c in item]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
np.save(args.output_file_path, integer_encoded)

