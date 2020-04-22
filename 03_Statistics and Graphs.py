#%%
import pandas as pd
import numpy as np
import glob
import os
import tables
import json
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%% 필요 파일 불러오기
CORE_data=pd.read_hdf('./CORE_data_ver3.h5',key='core')

with open('Auth_dict.json', 'r') as fp:
    Auth_dict = json.load(fp)
with open('Affil_dict.json', 'r') as fp:
    Affil_dict_abbr = json.load(fp)
with open('Country_dict.json', 'r') as fp:
    Country_dict = json.load(fp)

if not os.path.exists('./Figures'):
	os.makedirs('./Figures')
#%% 1. 연도별 논문 증가 그래프
# 2020년 이전 논문 출판 그래프
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['date'].dt.year[CORE_data['date'].dt.year<2020],zorder=3,palette=sns.color_palette("GnBu",20))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(i.get_text().replace('20',"'"))
ax.set_xticklabels(ax.get_xticklabels())
fig.set(ylabel='Number of Publications',xlabel = 'Year')
ax.grid(zorder=0, axis='y')

# 추가 정보 기술 필요시 사용
# ax.annotate('Hinton et al.,\n Science', xy=(0.33, 0.4),  xycoords='axes fraction',
#             xytext=(0.2, 0.6), textcoords='axes fraction',
#             arrowprops=dict(arrowstyle='simple'),
#             horizontalalignment='right', verticalalignment='top',
#             )
# ax.annotate('CNN won \n ILSVRC.', xy=(0.63, 0.41),  xycoords='axes fraction',
# xytext=(0.5, 0.6), textcoords='axes fraction',
# arrowprops=dict(arrowstyle='simple'),
# horizontalalignment='right', verticalalignment='top',
# )
# ax.annotate('AlphaGo match.', xy=(0.83, 0.57),  xycoords='axes fraction',
# xytext=(0.75, 0.75), textcoords='axes fraction',
# arrowprops=dict(arrowstyle='simple'),
# horizontalalignment='right', verticalalignment='top',
# )
plt.savefig('./Figures/publication_by_year.png',dpi=300, bbox_inches ='tight')

#%% 2. 1저자 기준 leading country 연도별 top 10국가별 키워드 증가 그래프 (2000-2019)
x=range(2000,2020) #2000년~2019년까지
y=[]
year_tick = ["'"+str(x)[2:] for x in range(2000,2020)]
## 상위 n 개국의 연도별 논문 현황 그래프 (기본 top 10)
n_rank = 10
order = pd.value_counts(CORE_data['fcountry']).iloc[:n_rank].index.to_list()
spec_set = CORE_data[CORE_data['fcountry'].isin(order)]
country_by_year = spec_set.groupby([spec_set['date'].dt.year,spec_set['fcountry']]).size().copy()
for country in order:
	y.append(country_by_year[:,country].reindex(x).fillna(0).to_list()) 
## Basic stacked area chart.
fig, ax = plt.subplots(1,1,figsize=(10,6))
plt.stackplot(x,y, labels=order,colors=sns.color_palette("Set3_r",10),zorder=3)
plt.legend(loc='upper left',ncol=2)
ax.set_xticks(x)
ax.set_xlim(2000,2019) #2000년~2019년까지
ax.set_ylim(0,300)
ax.set(ylabel= 'Number of Publications', xlabel = 'Year',xticklabels =year_tick)
ax.grid(zorder=0,axis='y')
plt.savefig('./Figures/publication_per_country_by_year.png',dpi=300,bbox_inches = 'tight')

#%%
##3. 1저자 기준 top n 랭킹
topN=15 # 저자 상위 n 명 (2000년~2020년까지)
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['fauth_id'],zorder=3,\
	order=CORE_data['fauth_id'].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.set_ylim(0,20)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'First authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publication_of_top_N_first_authors.png',dpi=300,bbox_inches = 'tight')

#%%
# 4. 1 저자 기준 top n 기관 랭킹 (2000년~2020년까지)
topN=15 # 기관 상위 n 개
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['faffil_id'],zorder=3,\
	order=CORE_data['faffil_id'].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Affil_dict_abbr[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax.set_ylim(0,20)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Affiliation')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publicaiton_of_top_N_first_affiliation.png',dpi=300,bbox_inches = 'tight')

#%%
# 5. 1저자 기준 top15 국가 랭킹
topN=15 # 국가 상위 n개
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['fcountry'],zorder=3,\
	order=CORE_data['fcountry'].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Country')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publicaion_of_top_N_first_country.png',dpi=300,bbox_inches = 'tight')


#%%
# 6. 공동 저술 포함 논문수 top N 랭킹 (저자)
topN=15
flat_auth_ids = pd.Series([ authid for authids in CORE_data.auth_ids for authid in authids])
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(flat_auth_ids,zorder=3,\
	order=flat_auth_ids.value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotat0ion=90)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publication_of_top_N_authors.png',dpi=300,bbox_inches = 'tight')

#%%
#7. 1저자 기준 top N 랭킹 (xx년이후)
topN=15 
fig, ax =plt.subplots(1,1,figsize=(10,6))
#최근 xx년 이후로 한정 (년도설정)
year_after = 2015
fig = sns.countplot(CORE_data['fauth_id'][CORE_data.date.dt.year>=year_after],zorder=3,\
	order=CORE_data['fauth_id'][CORE_data.date.dt.year>=year_after].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'First authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publication_of_top_N_first_authors_after_'+str(year_after)+'.png',dpi=300,bbox_inches = 'tight')

#%%
#8. 공저자 포함 top N 랭킹 (xx년이후)
topN=15
#최근 xx년 이후로 한정 (년도설정)
year_after = 2015
flat_auth_ids = pd.Series([ authid for authids in CORE_data.auth_ids[CORE_data.date.dt.year>=year_after] for authid in authids])
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(flat_auth_ids,zorder=3,\
	order=flat_auth_ids.value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/publication_of_top_N_authors_after_'+str(year_after)+'.png',dpi=300,bbox_inches = 'tight')

#%%
## 키워드 트렌드 그래프
# 특정 키워드를 포함하는 논문의 수를 연도별로 파악
core_keyword = ['artificial intelligence','machine learning','deep learning', 'data mining','neural network']
#2000~2019년까지
year_range=range(2000,2020)
keyword_trend_bag = {}
for keyword in core_keyword: 
    Query_title = (CORE_data['title'].str.contains(keyword,case=False,na=False,regex=True))
    Query_abstract = (CORE_data['abstract'].str.contains(keyword,case=False,na=False,regex=True))
    Query_keyword = (CORE_data['querykey'].str.contains(keyword,case=False,na=False,regex=True))
    Query = Query_title|Query_abstract|Query_keyword
    temp = CORE_data.date.dt.year.loc[Query].value_counts(sort=False)
    keyword_trend_bag[keyword] = temp.reindex(pd.Index(list(year_range))).fillna(0)

keyword_trend = pd.DataFrame(keyword_trend_bag)
keyword_trend['Total'] = keyword_trend.sum(axis=1)
keyword_trend['Year'] = keyword_trend.index
ax = keyword_trend.plot.bar(figsize=(10,6),y=core_keyword, width=1,stacked = False, legend=True,zorder = 3)

#ax2.set_ylim(ax.get_ylim())
keyword_trend.plot(y='Total',kind='line',ax=ax, use_index = False,\
    color='k', marker = 'o', legend=True,zorder = 3)
ax.set(ylabel = 'Number of Publications', xlabel = 'Year')
ax.grid(zorder=0,axis='y')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(i.get_text().replace('20',"'"))
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
# ax.set_xlim(-0.5,19.5)
plt.savefig('./Figures/publication_of_keyword_by_year.png',dpi=300,bbox_inches = 'tight')