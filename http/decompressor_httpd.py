from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM, Flatten, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
import models
import tempfile
import shutil


parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-batch_size', action='store', dest='batch_size', type=int,
                    help='model file')
parser.add_argument('-output', action='store', dest='output',
                    help='data file')
parser.add_argument('-input_file_prefix', action='store',dest='input_file_prefix',
                    help='compressed file name')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
args = parser.parse_args()
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
keras.backend.set_session(sess)

from keras import backend as K

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def create_data(rows, p=0.5):
        data = np.random.choice(2, rows, p=[p, 1-p])
        return data
 

def predict_lstm(len_series, timesteps, bs, alphabet_size, model_name, final_step=False):
        model = getattr(models, model_name)(bs, timesteps, alphabet_size)
        model.load_weights(args.model_weights_file)
        
        if not final_step:
                num_iters = int((len_series)/bs)
                series_2d = np.zeros((bs,num_iters), dtype = np.uint8)
                # open compressed files and decompress first few characters using
                # uniform distribution
                f = [open(args.temp_file_prefix+'.'+str(i),'rb') for i in range(bs)]
                bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
                dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]
                prob = np.ones(alphabet_size)/alphabet_size
                cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                cumul[1:] = np.cumsum(prob*10000000 + 1)                
                for i in range(bs):
                        for j in range(min(num_iters,timesteps)):
                                series_2d[i,j] = dec[i].read(cumul, alphabet_size)
                cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
                for j in (range(num_iters - timesteps)):
                        prob = model.predict(series_2d[:,j:j+timesteps], batch_size=bs)
                        cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
                        for i in range(bs):
                                series_2d[i,j+timesteps] = dec[i].read(cumul[i,:], alphabet_size)
                # close files
                for i in range(bs):
                        bitin[i].close()
                        f[i].close()
                return series_2d.reshape(-1)
        else:
                series = np.zeros(len_series, dtype = np.uint8)
                f = open(args.temp_file_prefix+'.last','rb')
                bitin = arithmeticcoding_fast.BitInputStream(f)
                dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
                prob = np.ones(alphabet_size)/alphabet_size
 
                cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                cumul[1:] = np.cumsum(prob*10000000 + 1)                
                for j in range(min(timesteps,len_series)):
                        series[j] = dec.read(cumul, alphabet_size)
                for i in (range(len_series-timesteps)):
                        prob = model.predict(series[i:i+timesteps].reshape(1,-1), batch_size=1)
                        cumul[1:] = np.cumsum(prob*10000000 + 1)
                        series[i+timesteps] = dec.read(cumul, alphabet_size)
                bitin.close()
                f.close()
                return series

def arithmetic_step(prob, freqs, dec):
        freqs.update_table(prob*10000000+1)
        return dec.read(freqs)

# variable length integer decoding http://www.codecodex.com/wiki/Variable-Length_Integers
def var_int_decode(f):
        byte_str_len = 0
        shift = 1
        while True:
                this_byte = struct.unpack('B', f.read(1))[0]
                byte_str_len += (this_byte & 127) * shift
                if this_byte & 128 == 0:
                        break
                shift <<= 7
                byte_str_len += shift
        return byte_str_len
def get_slice(data,flag):
    ind=np.where(np.array(data)==flag)[0]
    finaldata=[]
    for i in range(len(ind)):
        if i==0:
            finaldata.append(data[:ind[i]])
        else:
            finaldata.append(data[ind[i-1]+1:ind[i]])
    return finaldata
def translate(data,tmp_key=''):
    data_str=''
    for i in data:
        data_str=data_str+id2char_dict[str(i)]
    if tmp_key is not '' and tmp_key in re_keys.keys() and data_str.isnumeric() and len(data_str)<4:
        possible=range(len(re_keys[tmp_key]))
        possible=[ str(i) for i in possible ]
        if data_str in possible and re_keys[tmp_key][int(data_str)]!="**-1**":
            data_str=re_keys[tmp_key][int(data_str)]
    return data_str
import time

import time
import datetime
def translate_time(time_,mins):
    time_=int(translate(time_))+int(mins[0])
    strr=datetime.datetime.fromtimestamp(time_)
    year=strr.year
    month=strr.month
    day=strr.day
    hour=strr.hour
    minutes=strr.minute
    secons=strr.second
    if day<10:
        dat='0'+str(day)
    if hour<10:
        hour='0'+str(hour)
    if minutes<10:
        minutes='0'+str(minutes)
    if secons<10:
        secons='0'+str(secons)
    if month==9:
        month='Sep'
    return str(day)+'/'+str(month)+'/'+str(year)+':'+str(hour)+':'+str(minutes)+':'+str(secons)
def create_pid_block(data1,data2,mins):
    pid_key_list=['host-ip-adress']
    values=data1
    values=get_slice(values,0)
    blocks=data2
    flag=0
    common_key=[]
    data_processed=''
    blocks=get_slice(blocks,1)
    for index in range(len(blocks)):
        data_str='{'
        line=blocks[index]
        key=key_template_dict[0].split(',')
        line=line
        if flag==0:
            common_key=[val for val in key if val in pid_key_list]
            flag=1
        values2=get_slice(line,0)
        values2_begin=0
        len_it=len(key)
        for item_index in range(len_it):
            if key[item_index] in common_key:
                data_str=data_str+"\""+key[item_index]+"\":\""+translate(values[common_key.index(key[item_index])],tmp_key=key[item_index])+"\","
            else:
                if key[item_index]=='timestamp':
                    data_str=data_str+"\""+'timestamp'+"\":\""+str(translate_time(values2[values2_begin],mins))+"\","
                elif key[item_index]=='counter':
                    counters=int(translate(values2[values2_begin]))+int(mins[1])
                    if counters<1000:
                        counters=str(counters)
                        len_tmp=len(counters)
                        for i in range(4-len(counters)):
                            counters='0'+counters
                    data_str=data_str+"\""+'counter'+"\":\""+str(counters)+"\","
                else:
                    data_str=data_str+"\""+key[item_index]+"\":\""+translate(values2[values2_begin],tmp_key=key[item_index])+"\","
                values2_begin=values2_begin+1
        data_str=data_str[:len(data_str)-1]+"}"
        data_processed=data_processed+data_str
    return data_processed

def main():
        args.temp_dir = tempfile.mkdtemp()
        args.temp_file_prefix = args.temp_dir + "/compressed"
        tf.set_random_seed(42)
        np.random.seed(0)
        f = open(args.input_file_prefix+'.params','r')
        param_dict = json.loads(f.read())
        f.close()
        len_series = param_dict['len_series']
        batch_size = param_dict['bs']
        timesteps = param_dict['timesteps']
        mins=param_dict['mins']
        global id2char_dict
        global char2id_dict
        char2id_dict=param_dict['char2id_dict']
        id2char_dict = param_dict['id2char_dict']
        global key_template_dict
        key_template_dict=param_dict['key_template_dict']
        global re_values
        global re_keys
        re_values_dict=param_dict['re_values_dict']
        re_keys=re_values_dict[0]
        re_values=re_values_dict[1]
        key_template_dict=dict([val,key] for key,val in key_template_dict.items())
        f = open(args.input_file_prefix+'.combined','rb')
        for i in range(batch_size):
                f_out = open(args.temp_file_prefix+'.'+str(i),'wb')
                byte_str_len = var_int_decode(f)
                byte_str = f.read(byte_str_len)
                f_out.write(byte_str)
                f_out.close()
        f_out = open(args.temp_file_prefix+'.last','wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
        f.close()

        series = np.zeros(len_series,dtype=np.uint8)
        l = int(len_series/batch_size)*batch_size
        alphabet_size = len(id2char_dict)+3
        series[:l] = predict_lstm(l, timesteps, batch_size, alphabet_size, args.model_name)
        
        if l < len_series:
                series[l:] = predict_lstm(len_series - l, timesteps, 1, alphabet_size, args.model_name, final_step = True)
        np.save('x_decom',series)
        blocks=get_slice(np.append(series,3),2)
        data_processed_final='' 
        i=0
        flag=1
        while i < len(blocks)-2:
            i=i+1
            data_processed_final=data_processed_final+create_pid_block(blocks[i],blocks[i+1],mins)
            i=i+2
        with open(args.output,'w') as f:
            f.write(data_processed_final)

if __name__ == "__main__":
        main()

