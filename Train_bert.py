Train_NER_bert.py
未共享
类型
文本
大小
3 KB
占用的空间：
9 KB
位置
BERT-BiLSTM-CRF-NER
所有者
我
上次修改
下午4:51，我
上次打开时间
下午4:53，我自己
创建时间：
下午3:54，使用的应用：Google Drive Web
添加说明
查看者可以下载
from __future__ import print_function
import os
import argparse
import logging
import sys
import torch
from neuralnets.SentiBERTBiLSTMCNN import SentiBERTBiLSTMCNN
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.BERTWordEmbeddings import BERTWordEmbeddings
from keras import backend as K
import tensorflow as tf

import pandas as pd
data = pd.read_csv('../data.csv')
temp = []
for line in data['text']:
  line_temp = []
  for word in line.split(' '):
    if '@' not in word:
      line_temp.append(word)
  temp.append(' '.join(line_temp).replace('#','').replace('\n',' '))
data['text'] = temp



# :: Transform datasets to a pickle file ::
pickleFile = perpareDataset(data)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
sess = tf.Session(config=config)
K.set_session(sess)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


parser = argparse.ArgumentParser()
parser.add_argument("--bert_n_layers", type=int, default=2)
#parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_epochs", type=int, default=25)
parser.add_argument("--jobid", type=str, default="282NER_ru")
parser.add_argument("--dataset_name", type=str, default="282NER_ru")
parser.add_argument("--tagging_format", type=str, default="NER_IOB")
parser.add_argument("--embeddings_file", type=str, default="./embeddings/fastText157/cc.ru.300.vec.gz.top1.bin")
parser.add_argument("--bert_path", type=str, default="bert-base-multilingual-cased")
hp = parser.parse_args()
print("hyperparameters ",hp)

dataset_name = hp.dataset_name
embeddings_file = hp.embeddings_file
bert_path = hp.bert_path
jobid = hp.jobid
nepochs = hp.n_epochs
bert_n_layers = hp.bert_n_layers
tagging_format = hp.tagging_format


bert_mode = 'weighted_average'

if torch.cuda.is_available():
    print("Using CUDA")
    bert_cuda_device = 0
else:
    print("Using CPU")
    bert_cuda_device = -1

embLookup = BERTWordEmbeddings(embeddings_file, True, bert_path, bert_n_layers=bert_n_layers, bert_cuda_device=bert_cuda_device)

embLookup.cache_computed_bert_embeddings = True

bertfn = bert_path.split('/')
bertfn = bertfn[-1] if len(bertfn[-1]) > 1 else bertfn[-2]
embLookup.loadCache('embeddings/bert_'+bertfn+'_cache_'+dataset_name+'.pkl')
mappings, data = loadDatasetPickle(pickleFile)

params = {'classifier': ['CNN'], 'LSTM-Size': [100, 100], 'dropout': (0.5, 0.5)}

model = SentiBERTBiLSTMCNN(embLookup, params)
model.setMappings(mappings)
model.setDataset(datasets, data)
model.modelSavePath = "models/"+jobid+"/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=nepochs)