import streamlit as st
import advertools as adv
from advertools import crawl
import pandas as pd

st.sidebar.title("Internal Link Tool", anchor=None)
site = st.text_input("Add Your Website", max_chars=None, label_visibility="visible")

if site is not None:
	crawl(site, 'crawl.jl', follow_links=True)

crawl_df = pd.read_json('crawl.jl', lines=True)
st.table(crawl_df)
