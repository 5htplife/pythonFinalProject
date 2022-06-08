#house prices boston
#school rank
#fsq3rP25KpBZhbTUKC73zx2t5VFeZLi+yR0lk4zaq0xL1PQ= API

import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import glob
import os
import plotly.express as px
from scipy import stats
from bs4 import BeautifulSoup

import json
import requests
from urllib import request

from pandas.io.json import json_normalize
import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
#from geopy.distance import vincenty

from geopy.geocoders import (
    Nominatim,
)

@st.cache
def get_data():
    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/housing%202.csv")

st.set_page_config(
        page_title="Analyzing Housing Prices in California",
        page_icon="🧊",
        layout="centered"
    )

#DATA COLLECTION
#For a proper analysis, I will require the zipcodes of neighbourhoods in California

r=requests.get("https://www.zip-codes.com/state/ca.asp")
soup=BeautifulSoup(r.text)
new_dict={'zip':[], "city":[], "county":[], "type":[]}
soup1=soup.find("tr").find_all("tr")
for i in soup1[16:2600]:
    f=i.find_all("td")
    new_dict['zip'].append(f[0].text)
    new_dict["city"].append(f[1].text)
    new_dict['county'].append(f[2].text)
    new_dict['type'].append(f[3].text)
zipcodes = pd.DataFrame(new_dict)
zipcodes["Zip"]=zipcodes["zip"].str.split()
zipcodes["zip"]=[row["Zip"][2] for inx, row in zipcodes.iterrows()]
zipcodes=zipcodes.drop(columns="Zip")
zipcodes

#Also, I get geo data (latitude, latitude, zipcodes) in Notebook, so you can check it in the zip file (using API - not trivial by the way)

@st.cache
def get_lat_lon_zip():
    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/housing%202.csv")

#For proper visualization purposes I will require geojson, so I download it

#Now I obtain school rankings in California through scraping https://www.greatschools.org/california/schools/ website (using scrapy)
@st.cache
def get_school_rating():
    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/table.csv")

#Cool! Now we can work with the data

house_data = get_data()


