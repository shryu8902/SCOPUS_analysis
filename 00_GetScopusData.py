#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
from urllib3.exceptions import ProtocolError
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%% 저널_ISSN 데이터 받아오기
journal_list = pd.read_csv('Journal_ISSN.csv') #탐색하려는 저널의 ISSN 정보를 담은 파일 불러오기
exception_list = [] # 정보 읽어오는 중 에러가 발생한 저널 모음
for index, info in tqdm(journal_list.iterrows()):    
    h5_name = info.ISSN+'.h5'
    for year in range(2000,2010): # 논문 출판 연도 설정
        ISSN_Query = 'ISSN('+info.ISSN+')' # 저널 ISSN 매칭을 통해 특정 저널만 불러옴
        BASE_Query = ' AND DOCTYPE ( ar ) AND PUBYEAR = '+ str(year) # 문헌의 타입과 출판년도를 고정
        NUCLR_Query = ' AND TITLE-ABS-KEY (radioactive OR neutron OR nuclear OR reactor OR radiation)' #원자력 관련 키워드 포함 여부 확인
        if info.Category == 'OT' : # 저널 카테고리 구분에 따라 원자력 외 저널의 경우 원자력 키워드를 검색에 포함
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
        if not os.path.exists('./Scopus_Meta'):
            os.makedirs('./Scopus_Meta')
        temp.to_csv('./Scopus_Meta/'+info.ISSN+'_'+str(year)+'.csv')
pd.DataFrame(exception_list,columns = ['index','ISSN','year']).to_csv('./Exception_log.csv')
print('finish!!')