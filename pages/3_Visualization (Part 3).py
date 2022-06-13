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

    #def get_school_rating():
    #    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/table.csv")
    #school_data = get_school_rating()
    #school_data.rename(columns={'Address': 'zip'}, inplace=True)


    #def get_lat_lon_cities():
    #    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/lat_lon_cities.csv")


    #lat_lon_cities = get_lat_lon_cities()

    # Find the closest city to the houses
    from haversine import haversine, Unit

    #coordinates_for_cities = {}
    #for element in lat_lon_cities.iterrows():
    #    row = element[1]
    #    coordinates_for_cities[row['city']] = (row['lat'], row['lon'])

    #big_cities = ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco']  # top-4 in population
    #coordinates_for_big_cities = {key: value for key, value in coordinates_for_cities.items() if key in big_cities}


    #def find_closest_city(location, dicttttt):
    #    closest_city = ()
    #    for city in dicttttt.keys():
    #        distance = haversine(location, dicttttt[city])
    #        if closest_city == ():
    #            closest_city = distance
    #        while distance < closest_city:
    #            closest_city = distance
    #    return closest_city


    #geo_school = gpd.GeoDataFrame(school_data,
    #                                geometry=gpd.points_from_xy(school_data['Longitude'], school_data['Latitude']))

    #geodata_for_school = geodata1[['geometry', 'zcta']]

    #geo_school_merged = geo_school.sjoin(geodata_for_school, op="intersects", how="inner").drop(columns=['index_right'])

    def get_geo_school_merged():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/geo_school_merged.csv")

    geo_school_merged = get_geo_school_merged()
    geodata1 = get_geodata()

    st.write("The last map provides information about elementary school rankings in California neighbourhoods.")

    m1 = folium.Map([37.16611, -119.44944], zoom_start=6, tiles='openstreetmap')
    folium.Choropleth(
        geo_data=geodata1,
        name="choropleth",
        data=geo_school_merged,
        columns=["zcta", "Rating"],
        key_on='feature.properties.zcta',
        fill_color="OrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Elementary Schools in California",
    ).add_to(m1)

    map1 = st_folium(m1, key="fig2", width=700, height=700)