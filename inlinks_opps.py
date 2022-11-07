import streamlit as st
import advertools as adv
from advertools import crawl
import pandas as pd
from collections import OrderedDict
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

results = results[['url', 'title', 'links on page', 'total_inlinks', 'inlink score', 'links_url	']]

st.table(results)
