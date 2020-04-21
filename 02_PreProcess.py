#%%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re
#os.chdir('./proj_bib2')
#%% functions
def remove_amp(text):
    cleaned = re.sub('&amp;iacute;', 'i',str(text))
    cleaned = re.sub('&amp;oacute;','o',cleaned)
    cleaned = re.sub('&amp;','&',cleaned)
    return(cleaned)
def remove_copyright(abst,mode='forward'):
    if mode == 'forward':
    ## 마침표 없는 단어들 변경
        abst=abst.replace('Purpose: ','').replace('Background: ','').replace('Ltd ','Ltd. ').replace('CERN ','CERN. ')
        abst=abst.replace('The Authors ','The Authors. ').replace('The Author(s) ','The Author(s). ').replace('SUMMARY: ','')
        abst=abst.replace('Purpose:','').replace('Purpose : ', '').replace('Purpose. ','').replace('ABSTRACT:','')
        for i in range(2000,2021):
            abst = abst.replace('© '+str(i)+' ','© '+str(i)+'. ' )

    #해당 단어 포함시 문장 제외
    remove_words = ['© ', 'et al', 'Permissions','Published',\
        'Walter de Gruyter', ' Society', 'All right',' Ltd',' Pte', ' Inc', 'Elsevier','Springer','IEEE',\
        'Lippincott', 'Health Physics Society', 'Francis Group', 'Chinese Physical', 'The Author', 'Physical Society',\
        'Publishing Company', 'Limited', 'Oxford University Press', 'Sissa Medialab'] 
    #. 로 문장 쪼개기
    texts_list = abst.split('. ')
    # 첫번째부터 3번째 문장까지 위의 예외 단어 포함하는 최종 문장 번호 체크
    if mode == 'forward':
        pos = -1
        if len(texts_list) < 3:
            sentence_len = len(texts_list) 
        else:
            sentence_len = 3
        for j in range(sentence_len):
            if any(x in texts_list[j] for x in remove_words):

                pos = j
        if pos!=-1:
            abst_rev = '. '.join(texts_list[(pos+1):])
        else:
            abst_rev = abst
        abst_rev = re.sub('All right[s]? reserved[.|\s]','',abst_rev)
        return(abst_rev)
    elif mode =='backward':
    # 맨뒤쪽문장에서부터도 똑같이 최종 문장번호 체크
        neg_pos=0
        if len(texts_list) < 2 :
            sentence_len = len(texts_list)
        else: sentence_len = 2
        for j in range(sentence_len):
            if any(x in texts_list[-j-1] for x in remove_words):
                neg_pos = -j-1
        # 문장 인덱스 따라서 다시 abstract 이어붙이기.
        if neg_pos!=0:
            return('. '.join(texts_list[:neg_pos]))
        else:
            return(abst)
#%% 
CORE_meta = pd.read_csv('./ALL_meta_core_ver2.csv')
CORE_meta.reset_index(drop=True,inplace =True)
CORE_meta.affilname = CORE_meta.affilname.apply(lambda x: remove_amp(x) if type(x)==str else x)
#%%
#논문 Abstract 전처리
abstract = CORE_meta.description.apply(lambda x: remove_copyright(x,mode='backward')).apply(lambda x: remove_copyright(x,mode='forward'))

#특정 논문 abst 수정
len_abst = abstract.apply(lambda x: len(x))
#2615
abstract[2561]= CORE_meta.description[2561].split('. ')[1]
#2657
abstract[2603]= '. '.join(CORE_meta.description[2603].split('. ')[0:3])
#2815
abstract[2761]= CORE_meta.description[2761].split('. ')[1]
#3016
abstract[2962]= CORE_meta.description[2962].split('. ')[1]
#3038
abstract[2984]= CORE_meta.description[2984].replace('© 2018 ','')
#3202
abstract[3148]= CORE_meta.description[3148].replace('© 2005 Pleiades Publishing, Inc.','')
#3231
abstract[3177]= CORE_meta.description[3177].split('. ')[1]
#%%
## 기관 정보
## afid/ affil name 불일치 8건 : NaN 값이라 그럼. 
## 즉 afid가 주어져있으면 affil name도 무조건 들어가있음.
afid_split = CORE_meta.afid.apply(lambda x: x.split(';') if type(x)==str else x)
len_afid = CORE_meta.afid.apply(lambda x: len(x.split(';')) if type(x)==str else x)

affilname_split = CORE_meta.affilname.apply(lambda x: x.split(';') if type(x)==str else x)
len_affilname = CORE_meta.affilname.apply(lambda x: len(x.split(';')) if type(x)==str else x)

auth_afids_split = CORE_meta.author_afids.apply(lambda x: set(x.split(';')) if type(x)==str else x)
len_auth_afids = CORE_meta.author_afids.apply(lambda x: len(set(x.split(';'))) if type(x)==str else x)

## afid / country name 불일치 : 20 건 (nan 8건) -> 타기관 동일 국가에 해당하는 듯
country_split = CORE_meta.affiliation_country.apply(lambda x: x.split(';') if type(x)==str else x)
len_country = country_split.apply(lambda x: len(x) if type(x)==list else x)
affilname_split4dict = affilname_split[len_afid==len_country]
afid_split4dict = afid_split[len_afid==len_country]
country_split4dict = country_split[len_afid==len_country]

## affiliation dictionary 생성하기
Affil_dict = dict(zip([afid for afids in afid_split if type(afids)==list for afid in afids],[affil for affils in affilname_split if type(affils)==list for affil in affils]))
Country_dict = dict(zip([afid for afids in afid_split4dict for afid in afids],[country for countries in country_split4dict  for country in countries]))
Affil_dict.update({'':float('NaN')})
Country_dict.update({'':float('NaN')})
for i in list(set(Affil_dict.keys())-set(Country_dict.keys())):
    Country_dict.update({i:float('NaN')})
## 저자정보
## 전체저자 이름 수 / 전체저자 id 수 일치함
## 전체저자 이름 수 / 전제저자 afid 수 불일치 -> afids NaN값이라 그럼.
len_auth_names = CORE_meta.author_names.apply(lambda x: len(x.split(';')))
len_auth_id = CORE_meta.author_ids.apply(lambda x: len(x.split(';')))
len_auth_afid = CORE_meta.author_afids.apply(lambda x: len(x.split(';')) if type(x)==str else x)

## 저자 id - 이름 딕셔너리 생성
authid_split = CORE_meta.author_ids.apply(lambda x: x.split(';'))
authname_split = CORE_meta.author_names.apply(lambda x: x.split(';'))
Auth_dict = dict(zip([authid for authids in authid_split for authid in authids],[authname for authnames in authname_split for authname in authnames]))

#strings -> for statistics

f_author_id = authid_split.apply(lambda x: x[0])
f_author_name = f_author_id.apply(lambda x: Auth_dict[x])

f_affil_id = CORE_meta.author_afids.apply(lambda x: x.split(';')[0].split('-')[0] if type(x)!=float else x)
#afid_split.apply(lambda x: x[0] if type(x)==list else x)
f_affil_name = f_affil_id.apply(lambda x: Affil_dict[x] if type(x)!=float else x)   
f_affil_country = f_affil_id.apply(lambda x: Country_dict[x] if type(x)!=float else x)
journal = CORE_meta.publicationName
cite = CORE_meta.citedby_count
abstract = abstract
#date
date = pd.to_datetime(CORE_meta.coverDate)

#lists -> for networks
author_ids = authid_split
author_names = author_ids.apply(lambda x: [Auth_dict[j] for j in x]) 
affil_id = afid_split
affil_names = affil_id.apply(lambda x: [Affil_dict[j] for j in x] if type(x)==list else x)
countries = country_split.apply(lambda x: list(set(x)) if type(x)!=float else x)
#list - keyword
keywords = CORE_meta.authkeywords.apply(lambda x: x.split(' | ') if type(x)==str else x)

#%%
# SAVE Core information
#CORE_data = pd.DataFrame({'title':CORE_meta.title, 'eid':CORE_meta.eid, 'fauth_id':f_author_id, 'fauth_name':f_author_name,\
#    'faffil_id':f_affil_id, 'faffil_name':f_affil_name, 'fcountry':f_affil_country, 'auth_ids':author_ids,'auth_names':author_names, \
#    'affil_ids':affil_id, 'affil_names':affil_names,'country':countries, 'abstract':abstract, 'keyword':keywords, 'date':date, 'journal':journal, 'cite':cite}) 

#CORE_data.to_hdf('CORE_data_ver2.h5',key='core')

# SAVE dictionary data to JSON
#import json
#with open('Auth_dict_ver2.json', 'w') as fp:
#    json.dump(Auth_dict, fp)
#with open('Affil_dict_ver2.json', 'w') as fp:
#    json.dump(Affil_dict, fp)
#with open('Country_dict_ver2.json','w') as fp:
#    json.dump(Country_dict, fp)
#%%
# LOAD dictionary data from JSON

#with open('Auth_dict_ver2.json', 'r') as fp:
#    Auth_dict = json.load(fp)
#with open('Affil_dict_ver2.json', 'r') as fp:
#    Affil_dict = json.load(fp)
#with open('Country_dict_ver2.json', 'r') as fp:
#    Country_dict = json.load(fp)

#%% Dictionary preprocessing
#Abbr_dict_1 ={'Korea Atomic Energy Research Institute':'KAERI','Korea Advanced Institute of Science & Technology':'KAIST','Ulsan National Institute of Science and Technology' : 'UNIST', 'Korea Institute of Machinery & Materials':'KIMM'}
#Abbr_dict_2 ={'International':"Int'l",'[iI]nstitute':'Inst.','^Instit\w*':'Inst.','University':'Univ.','^Univer\w*':'Univ.','[N]ational':"Nat'l.",'[lL]aboratory':'Lab.', 'Chemistry':'Chem.','Center':'Cntr.','Centre':'Cntr.','Technology':'Technol.'}

#for key, value in Affil_dict.items():    
#    if type(value)==str:
#        for full, abbr in Abbr_dict_1.items():
#            value = re.sub(full, abbr, value)
#        for full, abbr in Abbr_dict_2.items():
#            value = re.sub(full,abbr, value)
#    Affil_dict[key]=value
#
#with open('Affil_dict_ver2_abbr.json', 'w') as fp:
#    json.dump(Affil_dict, fp)
