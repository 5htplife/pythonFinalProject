import streamlit as st
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
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
import folium
import folium.plugins as plugins
import geopandas as gpd

with st.echo(code_location="below"):
    st.write("## Machine Learning")

    st.write("As I mentioned above, in this section I will apply machine learning mechanisms: linear regression, xgboost and rainforest.")
    st.write("In the end, I will compare them based on mean squared error and r-squared error values.")

    @st.cache(allow_output_mutation=True)
    def get_coastline():
        url = "https://github.com/5htplife/FinalProject/raw/main/coastline.geojson"
        r = requests.get(url)
        coast = json.loads(r.text)
        return coast


    @st.cache(allow_output_mutation=True)
    def get_lat_lon_cities():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/lat_lon_cities")
    lat_lon_cities = get_lat_lon_cities() #download modified on the first page dataframe

    coast = get_coastline()
    coast_gpd = gpd.GeoDataFrame.from_features(coast['features'])
    coast_gpd['points'] = coast_gpd.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    coast_dict = coast_gpd.to_dict('records')
    a = [element['points'] for element in coast_dict]
    coast_list = [element for x in a for element in x]


    def find_closest_coast(location, listttttttt):
        closest_coast = ()
        for element in listttttttt:
            distance = haversine(location, element)
            if closest_coast == ():
                closest_coast = distance
            elif distance < closest_coast:
                closest_coast = distance
        return closest_coast


    # final_geo_data['coast_distance'] = final_geo_data.apply(
    #    lambda x: find_closest_coast((x['lon'], x['lat']), coast_tupless), axis=1).drop_duplicates()

    # This code takes a long time to run, so I just download the outcome: final_geo_data with column "coast_distance"

    #@st.cache()
    #def get_final_geo_data():
    #    return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/final_geo_data_with_coast_dist")

    #final_geo_data = get_final_geo_data()

    coordinates_for_cities = {}
    for element in lat_lon_cities.iterrows():
        row = element[1]
        coordinates_for_cities[row['city']] = (row['lat'], row['lon'])

    big_cities = ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco']  # top-4 in population
    coordinates_for_big_cities = {key: value for key, value in coordinates_for_cities.items() if key in big_cities}


    def find_closest_city(location, dicttttt):
        closest_city = ()
        for city in dicttttt.keys():
            distance = haversine(location, dicttttt[city])
            if closest_city == ():
                closest_city = (city, distance, dicttttt[city])
            while distance < closest_city[1]:
                closest_city = (city, distance, dicttttt[city])
        return closest_city


    #final_geo_data['closest_big_city'] = final_geo_data.apply(
    #    lambda x: find_closest_city((x['lat'], x['lon']), coordinates_for_big_cities), axis=1)
    #final_geo_data['closest_big_dist'] = [row["closest_big_city"][1] for inx, row in final_geo_data.iterrows()]

    #final_geo_data1 = final_geo_data.drop(columns=["city", "lat", "lon", "postal_code", "index_right", "closest_big_city"]).drop_duplicates()

    #For some reason this data gets mutated, but in Jupyter Notebook it works fine, so I converted it to csv and download now.

    @st.cache()
    def get_final_geo_data():
       return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/very_final_geo_data")

    final_geo_data1 = get_final_geo_data()

    def get_school_rating():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/table.csv")

    school_data = get_school_rating()
    school_data.rename(columns={'Address': 'zip'}, inplace=True)
    school_data_adj = school_data[['zip', 'Rating']]

    final_merged_data = final_geo_data1.merge(school_data_adj, left_on='zcta', right_on='zip',
                                             how='inner').dropna()

    final_merged_data = final_merged_data.drop(columns=['Unnamed: 0', 'geometry', 'zip']).drop_duplicates()

    final_merged_data.sort_values(by=['Rating'], ascending=False).reset_index(drop=True)
    final_data = final_merged_data.drop_duplicates(
        subset=['list_price', 'baths', 'beds', 'sqft', 'type', 'year_built']).drop(columns=['zcta'])

    final_data.type = final_data.type.map({"single_family": 1, "townhomes": 2, "condos": 3
                                              , "mobile": 4, "multi_family": 5, "duplex_triplex": 6,
                                           "coop": 7})

    final_data_wo_type = final_data.drop(columns=['type'])

    st.write("To my data, I have added the shortest distance to the coastline and to one of the biggest cities "
             "(Los Angeles, San Diego, San Jose, San Francisco).")

    st.write("Also, my dataset contains information about the type of houses for sale. So I code this categorical variable as following:"
             "single family houses - 1, townhomes - 2, condos - 3, mobile - 4, multi family houses - 5, duplex-triplex - 6, coop - 7.")

    st.write("So, now I can create a correlation matrix to see what influences the housing prices.")

    corr = final_data_wo_type.corr()

    fig1, ax1 = plt.subplots()
    x_axis_labels = ['baths', 'beds', 'house price', 'sqft', 'year built', 'population'
        , 'age (median)', 'white population', 'black afroamericans', 'hispanic'
           , 'coast distance', 'distance to the big city', 'Rating']
    y_axis_labels = ['baths', 'beds', 'house price', 'sqft', 'year built', 'population'
        , 'age (median)', 'white population', 'black afroamericans', 'hispanic'
           , 'coast distance', 'distance to the big city', 'Rating']
    sns.heatmap(data = corr, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='Spectral', annot_kws={"fontsize":6})
    ax1.set_title('Correlation Matrix')
    st.pyplot(fig1)

    st.write("Now, I want to check for multicollinearity in order to escape biases. If VIF (Variance Inflation Factor)"
             "is greater than 10, it indicates multicollinearity. In fact, anything greater than 4 is also a sign of multicollinearity.")

    ind_var = final_data[['tot_pop', 'baths', 'beds', 'type', 'sqft', 'year_built', 'age_median',
                          'white', 'black_afam', 'hispanic_l', 'coast_distance', 'closest_big_dist', 'Rating']]
    ind_var['intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data['var'] = ind_var.columns
    vif_data["VIF"] = [variance_inflation_factor(ind_var.values, i) for i in range(ind_var.shape[1])]
    vif = vif_data[vif_data['var'] != 'intercept']
    vif

    st.write("Since VIF is quite small for my dataset, I will keep all the variables.")
    st.write("I will also normalize all my independent variables.")

    def normalize(data):
        for element in range(len(data.columns) - 1):
            column = data.iloc[:, element]
            max_value = column.max()
            min_value = column.min()
            data.iloc[:, element] = (column - min_value) / (max_value - min_value)
        return data

    final_data1 = final_data.drop_duplicates()

    final_data_var = final_data1.drop(columns=["list_price"])

    final_data1['list_price'] = final_data1['list_price'].astype(int)

    st.write("## 1. Linear Regression")

    st.write("I run a linear regression with sklearn tools.")
    final_data_var_norm = normalize(final_data_var)

    x = final_data_var_norm
    y = final_data1[["list_price"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.67, random_state=42)
    x_trainn = x_train.to_numpy()
    y_trainn = y_train.to_numpy()
    x_testt = x_test.to_numpy()
    y_testt = y_test.to_numpy()

    st.write("I split the data into 80/20 - as it is a standard way of choosing a split.")
    linregr = linear_model.LinearRegression()
    linregress = linregr.fit(x_trainn, y_trainn)
    y_pred = linregress.predict(x_testt)

    sortim = y_pred.argsort(axis=0)
    y_predd = y_pred[sortim]
    y_testt = y_testt[sortim]
    y_predd = np.reshape(y_predd, -1)
    y_testt = np.reshape(y_testt, -1)

    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(y_predd)), y_testt, "ro", markersize=2, label="Actual house price")
    ax2.plot(np.arange(len(y_predd)), y_predd, "o", markersize=2, label="Predicted house price")
    ax2.set_ylim([-1e1, 1e7])
    ax2.legend()
    st.pyplot(fig2)

    mse = mean_squared_error(y_testt, y_predd)
    r2 = r2_score(y_testt, y_predd)
    st.write('MSE is {:.2f}'.format(mse))
    st.write('R-squared is {:.2f}'.format(r2))

    st.write("## 2. XGBoost")

    st.write("I run a linear regression with xgboost.")

    x = final_data_var_norm
    y = final_data1[["list_price"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42)
    x_trainn = x_train.to_numpy()
    y_trainn = y_train.to_numpy()
    x_testt = x_test.to_numpy()
    y_testt = y_test.to_numpy()

    grid_params = [{"max_depth": [6, 7, 8, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.3],
                    "n_estimators": [10, 30, 100]}]

    xgbreg = GridSearchCV(xgb.XGBRegressor(), grid_params, scoring="r2")
    xgbreg.fit(x_trainn, y_trainn)

    xgbreg.best_params_

    xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, objective='reg:squarederror')
    xgbbb = xgbreg.fit(x_trainn, y_trainn)
    y_pred = xgbbb.predict(x_testt)
    sortim = y_pred.argsort(axis=0)
    y_predd = y_pred[sortim]
    y_testt = y_testt[sortim]
    fig3, ax3 = plt.subplots()
    ax3.plot(np.arange(len(y_predd)), y_testt, "ro", markersize=2, label="actual price")
    ax3.plot(np.arange(len(y_predd)), y_predd, "o", markersize=2, label="predicted price")
    ax3.set_ylim([-1e1, 1e7])
    plt.legend()
    st.pyplot(fig3)

    mse = mean_squared_error(y_testt, y_predd)
    r2 = r2_score(y_testt, y_predd)
    st.write('MSE is {:.2f}'.format(mse))
    st.write('R-squared is {:.2f}'.format(r2))


    x = final_data_var_norm
    y = final_data[["list_price"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42)
    x_trainn = x_train.to_numpy()
    y_trainn = y_train.to_numpy()
    x_testt = x_test.to_numpy()
    y_testt = y_test.to_numpy()

    grid_params = [{"n_estimators": [10, 50, 100],
                    "max_depth": [6, 7, 8, 10],
                    "min_samples_leaf": [1, 2, 5]}
                   ]

    clf = GridSearchCV(RandomForestRegressor(), grid_params)
    clf.fit(x_trainn, y_trainn.ravel())

    clf.best_params_

    random_less = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2)

    les = random_less.fit(x_trainn, y_trainn.ravel())
    y_pred = les.predict(x_testt)
    sortim = y_pred.argsort(axis=0)
    y_predd = y_pred[sortim]
    y_testt = y_testt[sortim]
    fig4, ax4 = plt.subplots()
    ax4.plot(np.arange(len(y_predd)), y_testt, "ro", markersize=2, label="actual price")
    ax4.plot(np.arange(len(y_predd)), y_predd, "o", markersize=2, label="predicted price")
    ax4.set_ylim([-1e1, 1e7])
    plt.legend()
    st.pyplot(fig4)
    mse = mean_squared_error(y_testt, y_predd)
    r2 = r2_score(y_testt, y_predd)
    st.write('MSE is {:.2f}'.format(mse))
    st.write('R-squared is {:.2f}'.format(r2))


