import streamlit as st
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import seaborn as sns
import fiona
import requests
import folium.plugins as plugins


with st.echo(code_location="below"):
    @st.cache()
    def ask():
        url = "https://github.com/5htplife/pythonFinalProject/raw/master/zipcodes.geojson"
        return requests.get(url)


    def get_geodata():
        r = ask()
        geodata = json.loads(r.text)
        return geodata

    def get_final_mean_data():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/data_for_ml")

    final_geo_mean_data = get_final_mean_data()

    geodata1 = gpd.read_file("https://github.com/5htplife/pythonFinalProject/raw/master/zipcodes.geojson")

    #final_geo_mean_data['zcta'] = final_geo_mean_data['zcta'].astype('str')

    st.write("This visualization shows average housing prices in certain neighbourhoods, as well as provides"
             "some information about this neighbourhood (population, median age, % of white, black, hispanic population).")

    fig_mean = px.choropleth(final_geo_mean_data, locations='zcta', color='list_price',
                             hover_data=['list_price', 'tot_pop', 'age_median', 'white', 'black_afam', 'hispanic_l'],
                             geojson=geodata1, featureidkey='properties.zcta', projection="mercator",
                             labels={'list_price': 'House price in the area', 'tot_pop': 'Population in the area',
                                     'age_median': 'Median Age in the area',
                                     'white': 'White population (%)',
                                     'black_afam': 'Black American population (%)',
                                     'hispanic_l': 'Hispanic American population (%)'},
                             color_continuous_scale=[[0, 'rgb(253, 231, 37)'],
                                                     [0.05, 'rgb(94, 201, 98)'],
                                                     [0.1, 'rgb(33, 145, 140)'],
                                                     [0.2, 'rgb(59, 82, 139)'],
                                                     [1, 'rgb(68, 1, 84)']]
                             )
    fig_mean.update_geos(fitbounds="locations", visible=False)
    fig_mean.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_mean)