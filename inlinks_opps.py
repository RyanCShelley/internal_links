import streamlit as st
import advertools as adv
from advertools import crawl
import pandas as pd
from collections import OrderedDict
import numpy as np

st.set_page_config(layout="wide")
st.sidebar.title('Internal Links Tool')
st.sidebar.subheader('Add Your URL')
site = st.sidebar.text_input("Add Your Website", max_chars=None, label_visibility="visible")

if site is not None:
	crawl(site, 'crawl.jl', follow_links=True)

crawl_df = pd.read_json('crawl.jl', lines=True)

new_df = crawl_df[['url','title','links_url']]
new_df['links_url'] = new_df['links_url'].str.replace('@@',', ')
new_df = new_df.dropna(subset=['links_url'])
new_df['links_url'] = new_df['links_url'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
new_df['links on page'] = new_df['links_url'].str.split().str.len()
inlinks = new_df.links_url.str.split(expand=True).stack().value_counts()
inlinks_df = inlinks.to_frame()
inlinks_df.reset_index(inplace=True)
inlinks_df = inlinks_df.replace(',','', regex=True)
inlinks_df.columns = ["url","total_inlinks"]
inlinks_df = inlinks_df[inlinks_df["url"].str.contains(site)]
inlinks_df.sort_values(by=['total_inlinks'])
results = inlinks_df.merge(new_df)
results = results.drop_duplicates(subset=['url'])
rows = len(results.axes[0])

inlink_score = []

# Loop items in results
for link in results['total_inlinks']:
  score = link/rows
  inlink_score.append(score)
  
results["inlink score"] = inlink_score
results = results.sort_values(by='inlink score', ascending=False)
results = results[['url', 'title', 'links on page', 'total_inlinks', 'inlink score', 'links_url']]

st.header('Crawl Data')
st.dataframe(results)

@st.cache
def convert_df(results):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return results.to_csv().encode('utf-8')

csv = convert_df(results)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='internal_link_data.csv',
    mime='text/csv',
)

import plotly.express as px

fig = px.scatter(results, x="links on page", y="total_inlinks", color="inlink score",
                 size='inlink score', hover_data=['url'])
st.plotly_chart(fig, use_container_width=True)


network_df = results[['url','links_url']]
lst_col = 'links_url' 
x = network_df.assign(**{lst_col:network_df[lst_col].str.split(',')})

network = pd.DataFrame({
    col:np.repeat(x[col].values, x[lst_col].str.len())
    for col in x.columns.difference([lst_col])
    }).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]

network = network[network["url"].str.contains(site)]
network = network[network["links_url"].str.contains(site)]

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

GA = nx.from_pandas_edgelist(network, source="url", target="links_url")

color_map = []
for node in GA:
    if node in network["url"].values:
        color_map.append("blue")
    else: color_map.append("green") 


plt.figure(3, figsize=(50, 50))
nx.draw(GA, node_color=color_map, with_labels=False)
plt.show()
