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
CORE_data=pd.read_hdf('./CORE_data_ver3.h5',key='core')
#%% Read JSON files
with open('Auth_dict_ver2.json', 'r') as fp:
    Auth_dict = json.load(fp)
with open('Affil_dict_ver2_abbr.json', 'r') as fp:
    Affil_dict_abbr = json.load(fp)
with open('Country_dict_ver2.json', 'r') as fp:
    Country_dict = json.load(fp)

#%% 공저자 10인 이하 논문만 
Auth_count= CORE_data['auth_ids'].apply(lambda x: len(x) if type(x)==list else x)
author_connections = CORE_data[Auth_count<=10]['auth_ids'].apply(lambda x: list(combinations(x,2)))
author_connections_flat = [list(item) for sublist in author_connections for item in sublist]
for i,j in enumerate(author_connections_flat):
    if j[0]>j[1]:
        author_connections_flat[i]=[j[1],j[0]]
co_auth = pd.DataFrame(author_connections_flat,columns=['From','To']).groupby(['From','To']).size().reset_index()
co_auth.columns = ['From','To','Count']
G= nx.from_pandas_edgelist(co_auth,source='From',target='To',edge_attr='Count')

auth_ids_flat=pd.Series([afid for afids in CORE_data[Auth_count<=10]['auth_ids'] for afid in afids])
for n in set(auth_ids_flat):
	if n not in G.nodes():
		G.add_node(n)
nx.set_node_attributes(G,auth_ids_flat.value_counts().to_dict(),'Num_pub')
nx.set_node_attributes(G,CORE_data.fauth_id.value_counts().to_dict(),'Num_fpub')
nx.set_node_attributes(G,Auth_dict,'Name')
nx.write_gexf(G,'Authors_u10.gexf')

#%% Author Network


author_connections =CORE_data['auth_ids'].apply(lambda x: list(combinations(x,2)) if type(x)==list else x )
author_connections_flat = [list(item) for sublist in author_connections for item in sublist]
for i,j in enumerate(author_connections_flat):
    if j[0]>j[1]:
        author_connections_flat[i]=[j[1],j[0]]
co_auth = pd.DataFrame(author_connections_flat,columns=['From','To']).groupby(['From','To']).size().reset_index()
co_auth.columns = ['From','To','Count']
G= nx.from_pandas_edgelist(co_auth,source='From',target='To',edge_attr='Count')

auth_ids_flat=pd.Series([afid for afids in CORE_data['auth_ids'] for afid in afids])
for n in set(auth_ids_flat):
	if n not in G.nodes():
		G.add_node(n)
nx.set_node_attributes(G,auth_ids_flat.value_counts().to_dict(),'Num_pub')
nx.set_node_attributes(G,CORE_data.fauth_id.value_counts().to_dict(),'Num_fpub')
nx.set_node_attributes(G,Auth_dict,'Name')
nx.write_gexf(G,'Authors.gexf')

CORE_data.fauth_id.value_counts()

#%% Affiliation Network


affil_connections = CORE_data['affil_ids'].apply(lambda x: list(combinations(x,2)) if type(x)==list else x )
affil_connections_flat = [list(item) for sublist in affil_connections if type(sublist)== list for item in sublist]
for i,j in enumerate(affil_connections_flat):
    if j[0]>j[1]:
        affil_connections_flat[i]=[j[1],j[0]]

co_affil = pd.DataFrame(affil_connections_flat, columns = ['From','To']).groupby(['From','To']).size().reset_index()
co_affil.columns = ['From','To','Count']
G= nx.from_pandas_edgelist(co_affil,source='From',target='To',edge_attr='Count')

affil_ids_flat=pd.Series([afid for afids in CORE_data['affil_ids'] if type(afids)!=float for afid in afids])
for n in set(affil_ids_flat):
	if n not in G.nodes():
		G.add_node(n)
nx.set_node_attributes(G,affil_ids_flat.value_counts().to_dict(),'Num_pub')
nx.set_node_attributes(G,CORE_data.faffil_id.value_counts().to_dict(),'Num_fpub')
nx.set_node_attributes(G,Affil_dict_abbr,'Name')
#%% Country Network
country_connections = CORE_data['country'].apply(lambda x: list(combinations(x,2)) if type(x)==list else x )
country_connections_flat = [list(item) for sublist in country_connections if type(sublist)== list for item in sublist]
for i,j in enumerate(country_connections_flat):
    if j[0]>j[1]:
        country_connections_flat[i]=[j[1],j[0]]

co_country = pd.DataFrame(country_connections_flat, columns = ['From','To']).groupby(['From','To']).size().reset_index()
co_country.columns = ['From','To','Count']
G= nx.from_pandas_edgelist(co_country,source='From',target='To',edge_attr='Count')

country_flat=pd.Series([afid for afids in CORE_data['country'] if type(afids)!=float for afid in afids])
for n in set(country_flat):
	if n not in G.nodes():
		G.add_node(n)
nx.set_node_attributes(G,affil_ids_flat.value_counts().to_dict(),'Num_pub')
nx.set_node_attributes(G,CORE_data.faffil_id.value_counts().to_dict(),'Num_fpub')
nx.set_node_attributes(G,Affil_dict_abbr,'Name')

# %% Author network analysis
# 참고용;;;;;;;;;;;;;;
authors_per_paper = CORE_data['author'].to_list()
authids_per_paper = CORE_data['auth_ids'].to_list()

author_connections = CORE_data['auth_ids'][10]
authors_flat = pd.Series([author for authors in authors_per_paper for author in authors])
#authors_flat.value_counts().nlargest(10).plot(kind='bar')
co_auth = pd.DataFrame
author_connections = CORE_data['auth_ids'].apply(lambda x: list(combinations(x),2))

author_names = author_ids.apply(lambda x: [Auth_dict[j] for j in x]) 


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