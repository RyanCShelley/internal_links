import streamlit as st
import advertools as adv
from advertools import crawl
import pandas as pd
from collections import OrderedDict
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from rake_nltk import Rake
rake_nltk_var = Rake()
import trafilatura
import re
import nltk.corpus
nltk.download("stopwords")
from nltk.corpus import stopwords
from PIL import Image


st.set_page_config(layout="wide")

image = Image.open('cropped-Simplified-Search-Logo.png')

st.sidebar.image(image)
st.sidebar.title('Internal Links Tool')
st.sidebar.subheader('Add Your URL')
site = st.sidebar.text_input("Add Your Website", max_chars=None, label_visibility="visible")

if site is not None:
	crawl(site, 'crawl.jl', follow_links=True)

crawl_df = pd.read_json('crawl.jl', lines=True)

new_df = crawl_df[['url','title','links_url','body_text']]
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
results = results[['url', 'title', 'links on page', 'total_inlinks', 'inlink score', 'body_text','links_url']]

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



results = results.dropna(subset=['title'])
nltk_terms = []

# Loop items in results
for text in results['title']:
  rake_nltk_var.extract_keywords_from_text(text)
  keyword_extracted = rake_nltk_var.get_ranked_phrases()[:1]
  if keyword_extracted is not None: # assuming the download was successful
    nltk_terms.append(keyword_extracted)
  

results["keyword"] = nltk_terms

results['keyword'] = [','.join(map(str, l)) for l in results['keyword']]
results.keyword.str.split(expand=True).stack().value_counts()

keywords = results.keyword.str.split(expand=True).stack().value_counts()
keywords = keywords.to_frame()
keywords = keywords.reset_index()
keywords.columns = ["Keyword", "Frequnecy"]

fig = px.bar(keywords, x='Keyword', y='Frequnecy')
fig.show()

st.header('Keyword Frequency')

col1, col2 = st.columns([3, 1])

col1.plotly_chart(fig,use_container_width=True)

col2.dataframe(keywords)

st.header('Finding URLs with Low Inlink Score')


results = results.dropna()
stop_words = stopwords.words('english')
results['body_text'] = results['body_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

results.body_text = results.apply(lambda row: re.sub(r"http\S+", "", row.body_text).lower(), 1)
results.body_text = results.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.body_text.split())), 1)
results.body_text = results.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.body_text).split()), 1)


score = st.number_input('Set Max Inlink Score')


inlink_opps_score = results[results['inlink score'] < score]
inlink_opps_score = inlink_opps_score[~inlink_opps_score['url'].str.contains('category')]
inlink_opps_score = inlink_opps_score[~inlink_opps_score['url'].str.contains('author')]
inlink_opps_score = inlink_opps_score[~inlink_opps_score['url'].str.contains('tag')]

st.dataframe(inlink_opps_score, use_container_width=True)


st.header('Finding URLs to Link To')
target_url = st.text_input('What URL are you wanting to build links to')
target_keyword = st.text_input('What is the target keyword for the page you want to build links to')
target_keyword = target_keyword .lower()
st.text("""We only want to build inlinks from pages with a higher score than the page we are working on. So check the inlinks score of the page you are wanting to link from and add that number below""" )
page_inlink_score = st.number_input('Set URL Inlink Score')

inlink_ops = results[results['body_text'].str.contains(target_keyword)]
inlink_ops = inlink_ops[~inlink_ops['links_url'].str.contains(target_url)]
inlink_ops = inlink_ops[inlink_ops['inlink score'] > page_inlink_score]

inlink_ops = inlink_ops.drop(columns=['links_url','body_text'])


st.dataframe(inlink_ops, use_container_width=True)

@st.cache
def convert_df(inlink_ops):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return inlink_ops.to_csv().encode('utf-8')

csv = convert_df(inlink_ops)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='internal_link_ops.csv',
    mime='text/csv',
)
