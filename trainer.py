
import json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
# typical train
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger

import models

tf.set_random_seed(42)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-test', action='store', default=None,
                    dest='testdata',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-name', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')
parser.add_argument('-batchsize', action='store',
                    dest='batchsize',
                    help='batchsize')
parser.add_argument('-epoch', action='store',
                    dest='epoch',
                    help='epoch', default=4)
parser.add_argument('-param', action='store',dest='param')
import keras.backend as K

def loss_fn(y_true, y_pred):
        return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)
def generate_single_output_data(file_path,batch_size,time_steps):
        series = np.load(file_path)
        series = series.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(series)
        series = series.reshape(-1)
        return series,onehot_encoder
def process_data(serie,onehot_encoder,batch_size,time_steps):
    data = strided_app(serie, time_steps+1, 1)
    data1=np.copy(data)
    np.random.shuffle(data1)
    X = data1[:, :-1]
    Y = data1[:, -1:]
    Y = onehot_encoder.transform(Y)
    return X,Y
def generator2(serie,onehot_encoder,batchsize,sequence_length):
    num_batch=int((len(serie)-sequence_length+1)/batch_size)
    while True:
        choice=np.random.choice(range(sequence_length,len(serie)), len(serie)-sequence_length, replace=False)
        choice=choice[:int(len(choice)/batchsize)*batchsize]
        choice=choice.reshape(num_batch,batchsize)
        for i in range(num_batch):
            data=[]
            for j in choice[i]:
                data.append(serie[j-sequence_length:j+1])
            data=np.array(data)
            X_tmp = data[:, :-1]
            Y_tmp = data[:, -1:]
            Y_tmp = onehot_encoder.transform(Y_tmp)
            yield (X_tmp,Y_tmp)
from keras.utils.training_utils import multi_gpu_model
def fit_model(raw_data,onehot_encoder,bs, sequence_length,nb_epoch, model):
        optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
        model.compile(loss=loss_fn, optimizer=optim,metrics=['accuracy'])
        checkpoint = ModelCheckpoint(arguments.name, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)
        callbacks_list = [checkpoint, csv_logger, early_stopping]
        model.fit_generator(generator2(raw_data,onehot_encoder,bs,sequence_length),steps_per_epoch=int((len(raw_data)-sequence_length+1)/batch_size),epochs=nb_epoch, use_multiprocessing=True,shuffle=True, verbose=1, callbacks=callbacks_list)
                
                
arguments = parser.parse_args()
print(arguments)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
keras.backend.set_session(sess)

batch_size=int(arguments.batchsize)
sequence_length=64
num_epochs=int(arguments.epoch)
with open(arguments.param, 'r') as f:
    params = json.load(f)
alphabet_size = len(params['id2char_dict'])
raw_data,onehot_encoder= generate_single_output_data(arguments.data,batch_size, sequence_length)
model = getattr(models, arguments.model_name)(batch_size, sequence_length, alphabet_size)
fit_model(raw_data,onehot_encoder,batch_size,sequence_length,num_epochs,model)
