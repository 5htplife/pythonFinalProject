import matplotlib.pyplot as plt
import streamlit as st
import geopandas as gpd
import pandas as pd


with st.echo(code_location="below"):

    st.write("### School Rating Visualization")
    geodata1 = gpd.read_file("https://github.com/5htplife/pythonFinalProject/raw/master/zipcodes.geojson")


    def get_school_rating():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/table.csv")
    school_data = get_school_rating()
    school_data.rename(columns={'Address': 'zip'}, inplace=True)


    def get_lat_lon_cities():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/lat_lon_cities.csv")


    lat_lon_cities = get_lat_lon_cities()

    # Find the closest city to the houses
    from haversine import haversine, Unit

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
                closest_city = distance
            while distance < closest_city:
                closest_city = distance
        return closest_city


    geo_school = gpd.GeoDataFrame(school_data,
                                    geometry=gpd.points_from_xy(school_data['Longitude'], school_data['Latitude']))

    geodata_for_school = geodata1[['geometry', 'zcta']]

    geo_school_merged = geodata_for_school.sjoin(geo_school, op="intersects", how="inner").drop(columns=['index_right'])

    st.write("The last map provides information about elementary school rankings in California neighbourhoods.")
    st.write("Originally I created it with folium, but streamlit can't run it. That's why I recreated it with matplotlib"
             "Anyway, you can see it if you run the Jupyter Notebook file I attach.")

    geo_school_merged=geo_school_merged.dissolve(by="zcta", aggfunc='mean').reset_index(drop=True)

    geo_school_merged["Rating"]=geo_school_merged["Rating"].astype("float64")
    geo_school_merged["Rating"] = geo_school_merged["Rating"].fillna(0)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    fig, ax = plt.subplots(figsize=(10, 10))

    fig=geo_school_merged.plot(column='Rating', ax=ax, legend=True)
    plt.title("School Ratings across Regions")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend('School Rating')

    st.pyplot()

    #Original code:
    #m1 = folium.Map([37.16611, -119.44944], zoom_start=6, tiles='openstreetmap')
    #folium.Choropleth(
        #geo_data=geodata1,
        #name="choropleth",
        #data=geo_school_merged,
        #columns=["zcta", "Rating"],
        #key_on='feature.properties.zcta',
        #fill_color="OrRd",
        #fill_opacity=0.7,
        #line_opacity=0.2,
        #legend_name="Elementary Schools in California",
    #).add_to(m1)

    #st_folium(m1, key="fig2", width=700, height=700)