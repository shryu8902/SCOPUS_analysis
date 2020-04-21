#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
from urllib3.exceptions import ProtocolError
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval
os.chdir('/home/ashryu/proj_bib2')
#%% 저널_ISSN 데이터 받아오기
#from 2010 to 2020 / 3. 13일자
journal_list = pd.read_csv('Journal_ISSN.csv')
exception_list = []
for index, info in tqdm(journal_list.iterrows()):    
#    if index < 17: # 27번 데이터부터 다시 시행 / 39번부터 / 17번부터 ...
#        continue
    h5_name = info.ISSN+'.h5'
    for year in range(2000,2010):
        ISSN_Query = 'ISSN('+info.ISSN+')' 
        BASE_Query = ' AND DOCTYPE ( ar ) AND PUBYEAR = '+ str(year)
        NUCLR_Query = ' AND TITLE-ABS-KEY (radioactive OR neutron OR nuclear OR reactor OR radiation)'
        if info.Category == 'OT' :
            QUERY = ISSN_Query + BASE_Query + NUCLR_Query
        else :
            QUERY = ISSN_Query + BASE_Query
        print('number : {}, ISSN : {} Year : {}'.format(index, info.ISSN,year))        
        while True:
            try:
                s = ScopusSearch(QUERY)
            except ProtocolError:
                print("Connection Failure : Retry after 30 Seconds")
                time.sleep(30)
                error_info = [index,info.ISSN,year]
                exception_list.append(error_info)
                continue
            except OSError:
                print("Connection Failure : Retry after 30 Seconds")
                time.sleep(30)
                error_info = [index,info.ISSN,year]
                exception_list.append(error_info)
                continue            
            break
        temp=pd.DataFrame(s.results)
        temp.to_hdf('./'+h5_name,mode = 'a', key = str(year))
        temp.to_csv('./Scopus_Meta/'+info.ISSN+'_'+str(year)+'.csv')
pd.DataFrame(exception_list,columns = ['index','ISSN','year']).to_csv('./Exception_log.csv')
print('finish!!')

#%% h5 reader
#test=[]
#journal_list = pd.read_csv('Journal_ISSN.csv')

#for index, info in tqdm(journal_list.iterrows()):
#    if index > 3 :
#        continue
#    temp_read = pd.read_hdf('./Journal_Meta.h5',key = info.ISSN)
#    test.append(temp_read)
