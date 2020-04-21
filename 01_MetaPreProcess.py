#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import re
os.chdir('/home/ashryu/proj_bib2')
#%%
#저널 리스트 읽어오기
journal_list = pd.read_csv('Journal_ISSN.csv')
# 저널별 데이터 취합을 위한 dict 생성
journal_meta = {}
for index, info in tqdm(journal_list.iterrows()):
    #각 저널의 h5 파일 불러오기
    h5_name = './Scopus_Meta_h5/'+info.ISSN+'.h5'
    #temp 리스트에 연도별 데이터 프레임을 하나씩 저장
    temp=[]
    for year in range(2000,2021):
        df = pd.read_hdf(h5_name,key = str(year))
        if ~df.empty:
            temp.append(df)
    #temp 리스트 내 저장된 데이터 프레임 모두 합쳐서 저널별로 저장하기
    journal_meta[index] = pd.concat([temp[i] for i in range(len(temp))])
    journal_meta[index].issn = info.ISSN
    journal_meta[index].publicationName = info.Journal
# 모든 저널 데이터를 하나의 데이터 프레임으로 합치기
ALL_meta = pd.concat([journal_meta[i] for i in range(len(journal_list))],ignore_index=True)
ALL_meta = ALL_meta[~ALL_meta.creator.isnull()]
ALL_meta = ALL_meta[~ALL_meta.description.isnull()]
ALL_meta.reset_index(drop=True,inplace =True)
JEL_keyword = ALL_meta.authkeywords.str.contains('J[eE][lL]\sclassification|Classification\s[LNKSE][.]',case = True,na=False)                 
ALL_meta.authkeywords[JEL_keyword]=None
ALL_meta.coverDate = pd.to_datetime(ALL_meta.coverDate)
#%%
Category_ci = ['machine learning','deep learning','data mining', \
    'data[\s|-]driven','artificial intelligence',\
    'artificial intelligent', 'reinforcement learning','Q[\s|-]learning',\
    'supervised learning', 'unsupervised learning','clustering','regression',\
    'classification','learning algorithm', 'learning methodology']
#Category_cd = [' AI ']
Algorithm_ci = ['neural network','fully[\s|-]connected neural',\
    'feed[\s|-]forward neural','deep belief network'\
    'convolution neural','convolutional neural','recurrent neural',\
    'long short[\s|-]term memory',\
    'generative adversarial network','auto[\s|-]?encoder',\
    'variational autoencoder','support vector','random forest',\
    'gated recurrent unit','decision tree', 'natural language processing',\
    'xgboost', 'gradient boosting', 'restricted boltzmann machine',\
    'bayesian network', 'bayesian neural', 'k[\s|-]means', 'nearest neighbor',\
    'bagging']
Algorithm_cd = ['[\s|(]FNN','[\s|(]DNN', '[\s|(]CNN', '[\s|(]RNN', '[\s|(]LSTM', '[\s|(]VAE','[\s|(]SVM','[\s|(]SVR','[\s|(]GRU','[\s|(]DBN','[\s|(]RBM[^K\d]','[\s|(]k[\s|-]?NN']

Query_title = (ALL_meta.title.str.contains('|'.join(Category_ci),case=False,na=False) 
                | ALL_meta.title.str.contains('|'.join(Algorithm_ci),case=False,na=False)  
                | ALL_meta.title.str.contains('|'.join(Algorithm_cd),case=True,na=False))

Query_abstract = (ALL_meta.description.str.contains('|'.join(Category_ci),case=False,na=False) 
                | ALL_meta.description.str.contains('|'.join(Algorithm_ci),case=False,na=False)  
                | ALL_meta.description.str.contains('|'.join(Algorithm_cd),case=True,na=False))

Query_keyword = (ALL_meta.authkeywords.str.contains('|'.join(Category_ci),case=False,na=False) 
                | ALL_meta.authkeywords.str.contains('|'.join(Algorithm_ci),case=False,na=False)  
                | ALL_meta.authkeywords.str.contains('|'.join(Algorithm_cd),case=True,na=False))
Query = Query_title|Query_keyword|Query_abstract

#%%
#ALL_meta[Query].to_csv('ALL_meta_core_ver1.csv') # Discard.
ALL_meta[Query].to_csv('ALL_meta_core_ver2.csv') # JEL classification keyword 제거

# %%
test_Query = (ALL_meta.title.str.contains('automation|automated',case = False, na =False)
                |ALL_meta.description.str.contains('automation|automated',case = False, na =False)
                |ALL_meta.authkeywords.str.contains('automation|automated',case = False, na =False))
