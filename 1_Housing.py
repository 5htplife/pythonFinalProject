import streamlit as st
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from bs4 import BeautifulSoup
import requests
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import seaborn as sns
import fiona
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import haversine as hv
from haversine import haversine, Unit
from geopy.geocoders import (
    Nominatim,
)
import folium.plugins as plugins
import streamlit.components.v1 as components

with st.echo(code_location="below"):
    def get_main_data():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/house_prices_california2.csv")

    st.markdown("# Analyzing Housing Prices in California")
    st.write("## Data Collection Description")
    st.write("This project aims to provide a closer look at housing prices in California.")
    st.write("In order to do so, I used an open source Apify (https://apify.com/), where I gathered near 10'000 housing data items from https://www.realtor.com/")


    st.write("Some of the columns in the data frame contain null values.")

    st.write("That's why I drop the data this data items.")

    st.write("Also, I create a dataframe with mean values for each district for visualization and telegram-bot.")

    st.write("Then, I get the data of all zipcodes for each California city.")
    st.write("I do this by scraping the website https://www.zip-codes.com/state/ca.asp with Beautiful Soup.")

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
    zipcodes_cities=zipcodes.drop(columns="Zip")

    st.write("Since I don't have latitude and longitude, I get the data: zipcode, latitude, longitude using API (non-trivial!) from api.positionstack.com.")
    st.write("I attach Jupiter Notebook where I do this.")
    st.write("The file is uploaded on GitHub, as well.")

    @st.cache(allow_output_mutation=True)
    def get_lat_lon_zip():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/LatLonZIP.csv")

    lat_lon_zip = get_lat_lon_zip()
    lat_lon_zip['zip'] = lat_lon_zip['zip'].astype(int)
    lat_lon_zip['lat'] = lat_lon_zip['lat'].astype(float)
    lat_lon_zip['lon'] = lat_lon_zip['lon'].astype(float)
    zipcodes_cities["zip"] = zipcodes_cities['zip'].astype(int)
    lat_lon_cities = zipcodes_cities.merge(lat_lon_zip, on='zip')
    lat_lon_cities.drop(columns=['Unnamed: 0', 'type']).drop_duplicates()

    st.write("In order to get a full understanding of what influences the price, I obtain school rankings in California. ")
    st.write("Even if you visit Zillow https://www.zillow.com/, you will find out that they allow to filter by school ranking in the area, so this parameter is important in determinig the housing prices.")
    st.write("I get rankings by scraping https://www.greatschools.org/california/schools/For using Scrapy (for exact code see code below)")

    #Here is the code:
    #class SchoolSpider(scrapy.Spider):
    #    name = 'schoolspider'
    #   start_urls = ['https://www.greatschools.org/california/schools?page={}&view=table'.format(page) for page in
    #                  range(1, 240)]

    #    def parse(self, response):
    #        schools = json.loads(response.css('script::text').get().split(';')[20].replace('gon.search=', ''))[
    #            'schools']
    #        for school in schools:
    #           yield {'Name': school['name'],
    #                   'City': school['districtCity'],
    #                   'Latitude': school['lat'],
    #                   'Longitude': school['lon'],
    #                   'Address': school['zipcode'],
    #                   'Rating': school['rating'],
    #                   'SchoolType': school['schoolType']}

    st.write("The scraped csv file is uploaded on GitHub as well.")

    #@st.cache(allow_output_mutation=True)
    #def get_school_rating():
    #    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/table.csv")


    #download edited geojson from GitHub
    #@st.cache()
    #def ask():
    #    url = "https://github.com/5htplife/pythonFinalProject/raw/master/zipcodes.geojson"
    #    return requests.get(url)

    #def get_geodata():
    #    r = ask()
    #    geodata = json.loads(r.text)
   #     return geodata



    st.write("For visualization purposes I also download a json file with all California polygons.")
    st.write("The file was too large to upload on GitHub, so I edited it and uploaded on GitHub as well.")
    st.write("I attach a Jupiter Notebook code for the mentioned above procedure.")

    st.write("Finally, I obtain a json for the coordinates of the US coastline (I will need it to calculate distance between the coastline and the particular house).")


    @st.cache(allow_output_mutation=True)
    def get_coastline():
        url="https://github.com/5htplife/FinalProject/raw/main/coastline.geojson"
        r = requests.get(url)
        coast = json.loads(r.text)
        return coast


    st.write("Cool, now we can finally work with the data.")

    st.write("## Outline")
    st.write("### 1. Visualization")
    st.write("First, I will plot several maps to visualize what I have obtained.")
    st.write("For these purposes I will use plotly.express and folium maps.")
    st.write("### 2. Machine Learning")
    st.write("In this part of the project, I will use 3 machine learning methods: linear regression, xgboost and rainforest for house price prediction.")
    st.write("I will compare these methods and find the most accurate for my dataset.")
    st.write("### 3. Making Routes with Graphs")
    st.write("In this section, I will use NetworkX library and OSMnx libraries to make routes from all elementary schools in Beverly Hills to the nearest Starbucks Coffeshop.")
    st.write("In addition, I will plot routes from 50 elementary schools in San Francisco to the nearest bus stops.")
    st.write("### 4. Creating a Telegram-Bot")
    st.write("Finally, I will create Telegram-Bot which provides a user with the best elementary school and the average housing price in the given city.")
    st.write("### Remark (for an easier evaluation)")
    st.write("In this project I will use:"
             " pandas for data preprocessing; API (not trivial, harder than in home assignments) for data retrieving; "
             "BeautifulSoup (twice) and scrapy for web-scraping; visualization libraries (plotly.express, folium, seaborn, matplotlib);"
             " SQL and regex for creating a Telegram-Bot; geopandas, folium and shapely maaany times for almost everything; NetworkX and OSMnx"
             "for creating routes; numpy and scipy for creating routes and machine learning; XGBoost, RainForest and LinearRegression for machine learning. As for number of lines, definetely more than 120.")

    st.write("## Visualization")

    #df = get_main_data()
    #main_data = df[['baths', 'beds', 'city', 'lat', 'lon', 'list_price', 'sqft', 'type', 'postal_code', 'year_built']]

    #for column in main_data:
    #    if main_data[column].isnull().values.any() == True:
    #        print(column)

    #main_data_adj = main_data.dropna()

    #mean_main_data = main_data_adj.groupby('postal_code', as_index=False)['list_price'].mean()
    #mean_main_data = mean_main_data.astype(int)

    #mean_main_data_merged = mean_main_data.merge(lat_lon_cities, how='left', left_on='postal_code',
    #                                             right_on='zip').drop(columns=['type', 'Unnamed: 0']).drop_duplicates()

    #mean_main_data_ma = mean_main_data_merged.dropna(subset=['lat', 'lon'])
    #geo_mean_main_data_ma = gpd.GeoDataFrame(mean_main_data_ma, geometry=gpd.points_from_xy(mean_main_data_ma["lon"],
    #                                                                                        mean_main_data_ma['lat']))

    #geodata1 = get_geodata()
    #geodataa = gpd.GeoDataFrame.from_features(geodata1["features"])

    #geo_main_data = gpd.GeoDataFrame(main_data, geometry=gpd.points_from_xy(main_data["lon"], main_data['lat']))

    #final_geo_data = geo_main_data.sjoin(geodataa, op="intersects", how="inner")

    st.write("### Let's look at the housing prices.")

    st.write(" Black points on the map are particular houses.")

    st.write("I also marked 4 biggest cities in California (Los Angeles, San Francisco, San Diego, San Jose.")


    st.write("Below, you can see the code but Streamlit is toow weak for my geojson so I attach the Jupyter Notebook file for visualization.")

    filepathn = "Vizualization.html"
    HtmlFile = open(filepathn, 'r', encoding='utf-8')

    components.html(HtmlFile.read(), height=700, width=700)

    st.write("The second visualization provides information about elementary schools in California. On the 3rd page of visualization I have remade the map with matplotlib, but here is the original one.")
    #m = folium.Map([37.16611, -119.44944], zoom_start=6)
    #lat = final_geo_data.lat.tolist()
    #lon = final_geo_data.lon.tolist()
    #folium.Marker(
    #    location=[34.040587, -118.255403],
    #    popup="Los Angeles").add_to(m)

    #folium.Marker(
    #    location=[32.567022, -117.00425],
    #    popup="San Diego").add_to(m)

    #folium.Marker(
    #    location=[37.768106, -122.386927],
    #    popup="San Francisco").add_to(m)

    #folium.Marker(
    #    location=[37.335987, -121.777603],
    #    popup="San Jose").add_to(m)
    #HeatMap(list(zip(lat, lon))).add_to(m)
    #final_geo_data.apply(lambda x: folium.Circle(location=[x['lat'], x['lon']],
    #                                             radius=100, fill=True, color=x['type'], popup=x['list_price']).add_to(
    #    m), axis=1)
    #map = st_folium(m, key="fig1", width=700, height=700)

    #final_geo_mean_data = geo_mean_main_data_ma.sjoin(geodataa, op="intersects", how="inner").drop(
        #columns=['zip', 'index_right'])

    #final_geo_mean_data['white'] = (final_geo_mean_data['white'] / final_geo_mean_data['tot_pop'])
    #final_geo_mean_data['black_afam'] = (final_geo_mean_data['black_afam'] / final_geo_mean_data['tot_pop'])
    #final_geo_mean_data['hispanic_l'] = (final_geo_mean_data['hispanic_l'] / final_geo_mean_data['tot_pop'])
    #final_geo_mean_data = final_geo_mean_data.round(2)  #then I also save it as csv for Telegram-Bot

    #final_geo_mean_data['zcta'] = final_geo_mean_data['zcta'].astype('str')

#VISUALIZATION NOT RUN BY STREAMLIT
    #fig_mean = px.choropleth(final_geo_mean_data, locations='zcta', color='list_price',
                             #hover_data=['list_price', 'tot_pop', 'age_median', 'white', 'black_afam', 'hispanic_l'],
                             #geojson=geodata1, featureidkey='properties.zcta', projection="mercator",
                             #labels={'list_price': 'House price in the area', 'tot_pop': 'Population in the area',
                                     #'age_median': 'Median Age in the area',
                                     #'white': 'White population (%)',
                                     #'black_afam': 'Black American population (%)',
                                     #'hispanic_l': 'Hispanic American population (%)'},
                             #color_continuous_scale=[[0, 'rgb(253, 231, 37)'],
                                                     #[0.05, 'rgb(94, 201, 98)'],
                                                     #[0.1, 'rgb(33, 145, 140)'],
                                                     #[0.2, 'rgb(59, 82, 139)'],
                                                     #[1, 'rgb(68, 1, 84)']]
                             #)
    #fig_mean.update_geos(fitbounds="locations", visible=False)
    #fig_mean.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    #st.plotly_chart(fig_mean)






