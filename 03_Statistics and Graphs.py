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


if os.getcwd()!='/home/ashryu/proj_bib2':
	os.chdir('/home/ashryu/proj_bib2')
#os.chdir('./proj_bib')
# key_all does not include keywords by rake
#CORE_data=pd.read_hdf('Bib_data_in_list.h5',key='core')
# key_all include keywords by rake
#CORE_data=pd.read_hdf('./CORE_data.h5',key='core')
#CORE_data=pd.read_hdf('./CORE_data_ver2.h5',key='core')
# CORE_data=pd.read_hdf('./CORE_data_ver2.h5',key='core')
# CORE_data2=pd.read_hdf('./CORE_data_processed.h5',key='core')
# CORE_data['querykey']=CORE_data.keyword.apply(lambda x:'||'.join(x) if type(x)==list else x)
# CORE_data['key_pro']=CORE_data2.keywords_
# CORE_data.to_hdf('./CORE_data_ver3.h5',key='core')
CORE_data=pd.read_hdf('./CORE_data_ver3.h5',key='core')

#%% Read JSON files
with open('Auth_dict_ver2.json', 'r') as fp:
    Auth_dict = json.load(fp)
with open('Affil_dict_ver2_abbr.json', 'r') as fp:
    Affil_dict_abbr = json.load(fp)
with open('Country_dict_ver2.json', 'r') as fp:
    Country_dict = json.load(fp)
#
#%% 1. 연도별 논문 증가 그래프 (2000-2019)
# 
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['date'].dt.year[CORE_data['date'].dt.year<2020],zorder=3,palette=sns.color_palette("GnBu",20))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(i.get_text().replace('20',"'"))
ax.set_xticklabels(ax.get_xticklabels())
fig.set(ylabel='Number of Publications',xlabel = 'Year')
ax.grid(zorder=0, axis='y')
ax.annotate('Hinton et al.,\n Science', xy=(0.33, 0.4),  xycoords='axes fraction',
            xytext=(0.2, 0.6), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='simple'),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('CNN won \n ILSVRC.', xy=(0.63, 0.41),  xycoords='axes fraction',
xytext=(0.5, 0.6), textcoords='axes fraction',
arrowprops=dict(arrowstyle='simple'),
horizontalalignment='right', verticalalignment='top',
)
ax.annotate('AlphaGo match.', xy=(0.83, 0.57),  xycoords='axes fraction',
xytext=(0.75, 0.75), textcoords='axes fraction',
arrowprops=dict(arrowstyle='simple'),
horizontalalignment='right', verticalalignment='top',
)

#plt.savefig('./Figures/Fig1.png',dpi=300, bbox_inches ='tight')

#%% 2. 1저자 기준 leading country 연도별 top 10국가별 키워드 증가 그래프 (2000-2019)
x=range(2000,2020)
y=[]
year_tick = ["'"+str(x)[2:] for x in range(2000,2020)]
##top 10 국가 리스트
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
ax.set_xlim(2000,2019)
ax.set_ylim(0,300)
ax.set(ylabel= 'Number of Publications', xlabel = 'Year',xticklabels =year_tick)
ax.grid(zorder=0,axis='y')
plt.savefig('./Figures/Fig2.png',dpi=300,bbox_inches = 'tight')

#%%
#3. 1저자 기준 top 10 랭킹
topN=15
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
plt.savefig('./Figures/Fig3.png',dpi=300,bbox_inches = 'tight')

#%%
# 4. 1저자 기준 top15 기관 랭킹
topN=15
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
plt.savefig('./Figures/Fig4.png',dpi=300,bbox_inches = 'tight')


#%%
# 5. 1저자 기준 top15 국가 랭킹
topN=15
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['fcountry'],zorder=3,\
	order=CORE_data['fcountry'].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
#ax.set_ylim(0,20)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Country')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/Fig5.png',dpi=300,bbox_inches = 'tight')


#%%
# 6.공동 저술 포함 top 15랭킹
flat_auth_ids = pd.Series([ authid for authids in CORE_data.auth_ids for authid in authids])
topN=15
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(flat_auth_ids,zorder=3,\
	order=flat_auth_ids.value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax.set_ylim(0,20)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/Fig6.png',dpi=300,bbox_inches = 'tight')

#%%
#%%
#7. 1저자 기준 top 10 랭킹 (최근 15년이후)
topN=15
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(CORE_data['fauth_id'][CORE_data.date.dt.year>=2015],zorder=3,\
	order=CORE_data['fauth_id'][CORE_data.date.dt.year>=2015].value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'First authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/Fig7.png',dpi=300,bbox_inches = 'tight')

#%%
#8. 공저자 포함 top 15 랭킹 (15년이후)
flat_auth_ids = pd.Series([ authid for authids in CORE_data.auth_ids[CORE_data.date.dt.year>=2015] for authid in authids])
topN=15
fig, ax =plt.subplots(1,1,figsize=(10,6))
fig = sns.countplot(flat_auth_ids,zorder=3,\
	order=flat_auth_ids.value_counts().iloc[:topN].index,palette=sns.color_palette("GnBu_d",topN))
xtics = ax.get_xticklabels()
for i in xtics:
    i.set_text(Auth_dict[i.get_text()])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax.set_ylim(0,20)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.set(ylabel='Number of Publications',xlabel = 'Authors')
ax.grid(zorder=0, axis='y')
plt.savefig('./Figures/Fig8.png',dpi=300,bbox_inches = 'tight')

#%%
## 키워드 트렌드 그래프
#%% Keyword 1
core_keyword = ['artificial intelligence','machine learning','deep learning', 'data[\s|-]driven|data[\s|-]?mining','neural network']
#core_keyword = ['clustering','regression','classification','prediction']
#core_keyword = ['artificial[\s|-]neural[\s|-]network','fully[\s|-]connected[\s|-]neural[\s|-]network','feed[\s|-]?forward[\s|-]neural[\s|-]network','convolutional[\s|-]neural[\s|-]network','recurrent[\s|-]neural[\s|-]network','LSTM']
#core_keyword = ['generative adversarial network','bayesian neural network','graph neural network','transfer learning']
#core_keyword= ['decision tree','random forest','gradient boosting','support vector machine','support vector regression']
#core_keyword= ['critical heat flux','Departure from Nucleate Boiling Ratio|\sDNBR\s','Boiling heat[\s|-]?transfer']
keyword_trend_bag = {}
for keyword in core_keyword: 
    Query_title = (CORE_data['title'].str.contains(keyword,case=False,na=False,regex=True))
    Query_abstract = (CORE_data['abstract'].str.contains(keyword,case=False,na=False,regex=True))
    Query_keyword = (CORE_data['querykey'].str.contains(keyword,case=False,na=False,regex=True))
    Query = Query_title|Query_abstract|Query_keyword
    temp = CORE_data.date.dt.year.loc[Query].value_counts(sort=False)
    keyword_trend_bag[keyword] = temp.reindex(pd.Index(list(range(2000,2020)))).fillna(0)

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
ax.set_xlim(-0.5,19.5)
#%%
L=ax.legend(loc='upper left')
L.get_texts()[0].set_text('Total')
L.get_texts()[1].set_text('artificial intelligence')
L.get_texts()[2].set_text('machine learning')
L.get_texts()[3].set_text('deep learning')
L.get_texts()[4].set_text('data driven or data mining')
L.get_texts()[5].set_text('neural network')
plt.savefig('./Figures/key1.png',dpi=300,bbox_inches = 'tight')

#%% 추천받은 키워드

#%%


#core_keyword = ['machine learning', 'deep learning', 'reinforcement learning', 'artificial intelligence',\
#    'data mining | data.driven']
#core_keyword = ['clustering','regression','classification','prediction']
#core_keyword = ['artificial neural network', 'convolutional neural network',\
#    'recurrent neural network','random forest | decision tree','support vector machine | support vector regression']
#core_keyword = ['generative adversarial network','bayesian neural network','graph neural network','transfer learning']
core_keyword= ['artificial neural network','artificial neural networks']
temp_data = {}
for keyword in core_keyword: 
    Query_title = (data['title'].str.contains(keyword,case=False,na=False,regex=True))
    Query_abstract = (data['abstract'].str.contains(keyword,case=False,na=False,regex=True))
    Query_keyword = (data['querykey'].str.contains(keyword,case=False,na=False,regex=True))
    Query = Query_title|Query_keyword|Query_abstract

    temp = data.date.dt.year.loc[Query].value_counts(sort=False)
    temp_data[keyword] = temp.reindex(pd.Index([2015, 2016, 2017, 2018, 2019, 2020]))

test = pd.DataFrame(temp_data)
test['Total'] = test.sum(axis=1)
test['Year'] = test.index
ax = test.plot.bar(y=core_keyword, stacked = False, legend=True,zorder = 3)
#ax2.set_ylim(ax.get_ylim())
test.plot(y='Total',kind='line',ax=ax, use_index = False,\
    color='k', marker = 'o', legend=True,zorder = 3)
ax.legend(loc='upper left')

ax.set(ylabel = 'Number of Publications', xlabel = 'Year')
ax.grid(zorder=0,axis='y')
ax.set_xlim(0,20)


#%%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
CORE_data.groupby([CORE_data['date'].dt.year,CORE_data['date'].dt.quarter]).size().plot(kind='bar')
CORE_data['date'].dt.year.value_counts(sort=False).plot(kind='bar')
#%% 1. 연도별 논문 증가량
#%% 2. 분기별 논문 증가량
quarterly = CORE_data.groupby([CORE_data['date'].dt.year, CORE_data['date'].dt.quarter])
fig = sns.countplot(CORE_data.groupby([CORE_data['date'].dt.year,CORE_data['date'].dt.quarter]))
fig.set(ylabel = 'Number of Publications', xlabel = 'Year')



#%% 3. 전체 기간 퍼블리케이션 랭킹 by country and by year?
n_rank = 10
order = pd.value_counts(CORE_data['fcountry']).iloc[:n_rank].index.to_list()
spec_set = CORE_data[CORE_data['fcountry'].isin(order)].groupby(['fcountry',]).size()
country_by_year = spec_set.groupby([spec_set['date'].dt.year,spec_set['fcountry']]).size().copy()
#country_by_year = spec_set['f_country'].groupby([spec_set['date'].dt.year,spec_set['f_country']]).count()
colors = sns.color_palette()
margin_bottom = np.zeros(n_rank)
fig, ax = plt.subplots(1, 1)
for num, year in enumerate(range(2000,2021)):
	values = country_by_year[year].reindex(order).copy()
	values.plot.bar(stacked=True,bottom=margin_bottom,color=colors[num],zorder=3)
	#country_by_year[year].sort_values(ascending=False).plot.bar(stacked=True,bottom=margin_bottom,color=colors[num])
	margin_bottom +=values
ax.set(ylabel = 'Number of Publications', xlabel = 'Country')
ax.grid(zorder=0,axis='y')
ax.legend(range(2015,2021))
#plt.savefig('Pub_Over_Country_t10.png',dpi=300,bbox_inches = 'tight')
#%%
x=range(2000,2020)
y=[]
year_tick = [str(x) for x in list(x)]
for country in order:
	y.append(country_by_year[:,country].reindex(x).to_list()) 
# Basic stacked area chart.
fig, ax = plt.subplots(1,1)
plt.stackplot(x,y, labels=order)
plt.legend(loc='upper left',ncol=2)
ax.set_xticks(x)
ax.set_ylim(0,400)
ax.set(ylabel= 'Number of Publications', xlabel = 'Year',xticklabels =year_tick)
ax.grid(zorder=0,axis='y')
#plt.savefig('Pub_Over_Year_by_Country_t10.png',dpi=300,bbox_inches = 'tight')


#%% 4. 전체 기간 퍼블리케이션 랭킹 by affiliation
#TODO : 년도가 없는 경우는 없음
n_rank=10
order = pd.value_counts(CORE_data['f_affil']).iloc[:n_rank].index
order_list =order.to_list()
order_tick = [x[:30] + (x[30:] and '...') for x in order_list ]
spec_set = CORE_data[CORE_data['f_affil'].isin(order_list)]
affil_by_year = spec_set.groupby([spec_set['date'].dt.year, spec_set['f_affil']]).size().copy()
colors = sns.color_palette()
margin_bottom = np.zeros(n_rank)
fig, ax = plt.subplots(1,1)
for num, year in enumerate(range(2015,2021)):
	values = affil_by_year[year].reindex(order).copy()
	values[np.isnan(values)]=0
	values.plot.bar(stacked=True,bottom=margin_bottom,color=colors[num],zorder=3)
	margin_bottom +=values
ax.set(ylabel = 'Number of Publications', xlabel = 'Affiliation', xticklabels =order_tick)
ax.grid(zorder=0,axis='y')
ax.legend(range(2015,2021))
plt.savefig('Pub_Over_Affil_t10.png',dpi=300,bbox_inches = 'tight')

#%% 5. 전체 기간 퍼블리케이션 by journal
n_rank=11
order = pd.value_counts(CORE_data['journal']).iloc[:n_rank].index
order_list = order.to_list()
order_tick = [x[:30] + (x[30:] and '...') for x in order_list ]
spec_set = CORE_data[CORE_data['journal'].isin(order_list)]
journal_by_year = spec_set.groupby([spec_set['date'].dt.year, spec_set['journal']]).size().copy()
colors = sns.color_palette()
margin_bottom = np.zeros(n_rank)
fig, ax = plt.subplots(1,1)
for num, year in enumerate(range(2015,2021)):
	values = journal_by_year[year].reindex(order).copy()
	values[np.isnan(values)]=0
	values.plot.bar(stacked=True,bottom=margin_bottom,color=colors[num],zorder=3)
	margin_bottom +=values
ax.set(ylabel = 'Number of Publications', xlabel = 'Journal title', xticklabels =order_tick)
ax.grid(zorder=0,axis='y')
ax.legend(range(2015,2021))
plt.savefig('Pub_Over_Journal_t11.png',dpi=300,bbox_inches = 'tight')
#%% 6. 
spec_set2 = data[data.journal.isin(order_list)]
temp = round(spec_set.journal.value_counts() / spec_set2.journal.value_counts() *100,2).to_frame().copy()
## should check the order of data
temp.columns=['Publication Ratio']
temp['Impact Factor']= pd.Series([1.380, 1.343, 1.457, 1.428, 2.547, 1.186, 1.541, 1.546, 1.433, 3.132, 4.162], index = temp['Publication Ratio'].index) #%%
#%%
ax = temp['Publication Ratio'][order].plot(kind = 'bar',zorder=3,color=sns.color_palette("GnBu_d",11))
ax1 = ax.twinx()

temp['Impact Factor'][order].plot(zorder=4,ax=ax1,color='r',marker ='o')
ax.set(xlabel = 'Journal title',ylabel = 'AI publication ratio (%)',xticklabels = order_tick)
ax.grid(zorder=0, axis='y')
ax1.set_xlim(-0.5,10.5)
ax1.set(ylabel = 'Impact factor (2018)',xticklabels=order_tick)
ax.legend(labels='Publciation Ratio',loc='best')
ax.legend()
ax1.legend()
plt.savefig('Journal_t11_IF_and_ratio.png',dpi=300,bbox_inches = 'tight')
#%% 7. 논문 저자 랭킹
CORE_data.groupby(['fcountry']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['fauth_name']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['f_affil']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['journal']).size().nlargest(10).plot(kind='bar')

#%%7.  연도별 키워드 워드클라우드
from wordcloud import WordCloud, STOPWORDS
key_by_year = pd.DataFrame(CORE_data.key_all.groupby([CORE_data['date'].dt.year]))
authkey_by_year = pd.DataFrame(CORE_data.key_auth.groupby([CORE_data['date'].dt.year]))
tfidf_by_year = pd.DataFrame(CORE_data.key_abst_tfidf.groupby([CORE_data['date'].dt.year]))

authkeyword_by_year = {}
for i,j in enumerate(range(2015,2021)):
	print(i,j)
	temp = [x for lists in authkey_by_year.iloc[i,1].ravel() for x in lists]
	authkeyword_by_year[i]=('; ').join(temp)

tfidfkeyword_by_year = {}
for i,j in enumerate(range(2015,2021)):
	print(i,j)
	temp = [x for lists in tfidf_by_year.iloc[i,1].ravel() for x in lists]
	tfidfkeyword_by_year[i]=('; ').join(temp)

keyword_by_year = {}
for i,j in enumerate(range(2015,2021)):
	print(i,j)
	temp = [x for lists in key_by_year.iloc[i,1].ravel() for x in lists]
	keyword_by_year[i]=('; ').join(temp)
#%% 
wordcloud = WordCloud(background_color='white', max_font_size=100).generate(tfidfkeyword_by_year[4])
fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#%% 8. 
from collections import Counter
Counter(temp)


#%% 전체 키워드 워드 클라우드
temp = [x for lists in CORE_data.key_all.ravel() for x in lists]
keyword_flat = ('; ').join(temp)
new_stop = STOPWORDS
new_stop.update(Category_ci)
new_stop.update(Algorithm_ci)
new_stop.update(['using','Neural Network','Neural networks','Neural Network','artificial neural','based','method','two'])

wordcloud = WordCloud(background_color='white', max_font_size=100,stopwords=new_stop).generate(keyword_flat)
fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#%%


#%%
fig = sns.countplot('f_affil',data = CORE_data,order=order)
plt.xticks(rotation=90)
fig.set(ylabel = 'Number of Publications', xlabel = 'Affiliation')



sns.barplot(CORE_data['date'].dt.year.value_counts(sort=False))


CORE_data.groupby(['f_country']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['f_author']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['f_affil']).size().nlargest(10).plot(kind='bar')
CORE_data.groupby(['journal']).size().nlargest(10).plot(kind='bar')

# %% Author network analysis
authors_per_paper = CORE_data['author'].to_list()
authors_flat = pd.Series([author for authors in authors_per_paper for author in authors])
#authors_flat.value_counts().nlargest(10).plot(kind='bar')

author_connections = list(map(lambda x: list(combinations(x, 2)), authors_per_paper))
author_connections = [item for sublist in author_connections for item in sublist]
co_auth = pd.DataFrame(author_connections, columns = ['From','To'])
co_auth = co_auth.groupby(['From','To']).size().reset_index()
#create edges
co_auth.columns=['From','To','Count'] 
#create graph from edges
G=nx.from_pandas_edgelist(co_auth,source='From',target='To',edge_attr='Count')
#add nodes based on the number of publications
for n in set(authors_flat):
	if n not in G.nodes():
		G.add_node(n)
nx.set_node_attributes(G,authors_flat.value_counts().to_dict(),'Num_pub')

nx.write_gexf(G,'Authors_cowork_nx.gexf')

topNnodes = [n for n in list(G.nodes()) if n in authors_flat.value_counts().nlargest(100)]
G_topN = G.subgraph(topNnodes)
nx.write_gexf(G_topN,'Authors_cowork_subnx.gexf')

#%% keyword / wordcloud
## Word Cloud!!!!
# CORE_data.key_all

import math
from wordcloud import WordCloud

keywords_per_paper = CORE_data.key_all.to_list()
keywords_flat = []
for keywords in keywords_per_paper:
	for keyword in keywords:
		if type(keyword) != float:
			keywords_flat.append(keyword)

keywords_flat_to_text = ('; ').join(keywords_flat)
wordcloud = WordCloud(background_color='white', max_font_size=100).generate(keywords_flat_to_text)
fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")


#%%
# TODO : Keyword coocurrence network!!

keywords_per_paper = CORE_data['key_all']



fig = plt.figure(figsize=(10,10))
#pos = nx.spring_layout(G_topN, iterations=20)
pos = nx.kamada_kawai_layout(G_topN)
nx.draw(G_topN, pos,width = 1, with_labels=True)



test = authors_per_paper[0]
author_connections = list(map(lambda x: list(cominbations(x[]))))


test = CORE_data.groupby(CORE_data['date'].dt.year)
list(test)

# %%