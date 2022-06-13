import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import geonetworkx as gnx
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import cKDTree

with st.echo(code_location="below"):

    st.write("## Creating routes")

    st.write("Since we have a dataset with elementary schools in California, I have decided to create route paths.")

    st.write("### Where is Starbucks, Suzie?")

    st.write("If you have a warning that the page doesn't respond wait for a little bit, please.")
    st.write("This map shows routes between elementary schools in Beverly Hills and the nearest Starbucks Coffeeshop to each of them.")
    st.write("I use a local html file created from running this code for you not to wait for a long time.")
    st.write("This map also shows every walking route in Beverly Hills.")

    @st.cache(allow_output_mutation=True)
    def get_school_data():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/school_data")

    school_data = get_school_data()

    #geo_school = gpd.GeoDataFrame(school_data,
    #                              geometry=gpd.points_from_xy(school_data['Longitude'], school_data['Latitude']))
    #geo_school_bv = geo_school[geo_school.City
    #                           == "Beverly Hills"]
    #geo_school_bv = geo_school_bv[['geometry']]

    #place = "Beverly Hills, California, USA"
    #tags = {"amenity": "cafe"}
    #G = ox.graph_from_place(place, network_type="walk")

    #gdf = ox.geometries_from_place(place, tags)
    #gdf_star = gdf[gdf['official_name'] == "Starbucks Coffee"]
    #df_starr = gdf_star[['geometry']]


    # adapted the code for this funcation from here:
    # https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    def ckdnearest(gdf1, gdf2):
        gdf1_to_np = np.array(list(gdf1['geometry'].apply(lambda geom: (geom.x, geom.y))))
        gdf2_to_np = np.array(list(gdf2['geometry'].apply(lambda geom: (geom.x, geom.y))))
        btree = cKDTree(gdf2_to_np)
        dist, idx = btree.query(gdf1_to_np, k=1)
        gdf2_nearest = gdf2.iloc[idx]
        return gdf2_nearest


    #a = ckdnearest(geo_school_bv, gdf_starr)
    #a1 = a[['geometry']]

    #nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    #edges = ox.graph_to_gdfs(G, edges=True, nodes=False)

    #nodes = nodes.append(a, ignore_index=True)
    #G2 = ox.graph_from_gdfs(nodes, edges)

    #nn = []

    #for x in a.geometry:
    #    lon = x.x
    #    lat = x.y
    #    n = ox.distance.nearest_nodes(G, lon, lat)
    #    nn.append(n)

    #nn1 = []

    #for x in geo_school_bv.geometry:
    #    lon = x.x
    #    lat = x.y
    #    n1 = ox.distance.nearest_nodes(G, lon, lat)
    #    nn1.append(n1)

    #routes = []
    #for count, element in enumerate(nn):
    #    origin_node = element
    #    destination_node = nn1[count]
    #    route = nx.shortest_path(G, origin_node, destination_node)
    #    routes.append(route)


    #def route_plot(x):
    #    m1 = ox.plot_graph_folium(G2, popup_attribute="name", weight=2, color="#8b0000", tiles='openstreetmap',
    #                              opacity=0.5)
    #    if x == 0:
    #        return ox.plot_route_folium(G2, routes[0], route_map=m1, weight=5)
    #    else:
    #        return ox.plot_route_folium(G2, routes[x], route_map=route_plot(x - 1), weight=5)


    #map1 = route_plot(4) for time efficiency I use a plotted file (for you not to wait for a bit)
    filepath = "map1.html"
    #map1.save(filepath)

    HtmlFile = open(filepath, 'r', encoding='utf-8')

    components.html(HtmlFile.read(), height=500, width=700)


    #place1 = "San Francisco, USA"
    #tags1 = {"highway": "bus_stop"}
    #G_sf = ox.graph_from_place(place1, network_type="drive")

    #m_sf = ox.plot_graph_folium(G_sf, popup_attribute="name", weight=2, color="#8b0000", tiles='openstreetmap',
    #                            opacity=0.5)

    #geo_school = gpd.GeoDataFrame(school_data,
    #                              geometry=gpd.points_from_xy(school_data['Longitude'], school_data['Latitude']))
    #geo_school_sf = geo_school[geo_school.City
    #                           == "San Francisco"]
    #geo_school_sf = geo_school_sf[['geometry']]

    #gdf_sf = ox.geometries_from_place(place1, tags1)

    #a_sf = ckdnearest(geo_school_sf, gdf_sf)
    #a_sff = a_sf[['geometry']]

    #nn1_sf = []

    #for x in a_sff.geometry:
    #    lon = x.x
    #    lat = x.y
    #    n_sf1 = ox.distance.nearest_nodes(G_sf, lon, lat)
    #    nn1_sf.append(n_sf1)

    #nn2_sf = []

    #for x in geo_school_sf.geometry:
    #    lon = x.x
    #    lat = x.y
    #    n_sf2 = ox.distance.nearest_nodes(G_sf, lon, lat)
    #    nn2_sf.append(n_sf2)

    #routes_sf = []

    #for count, element in enumerate(nn1_sf):
    #    origin_node = element
    #    destination_node = nn2_sf[count]
    #    route_sf = nx.shortest_path(G_sf, origin_node, destination_node)
    #   if len(route_sf) > 1:
    #       routes_sf.append(route_sf)


    #def route_plot_sf(x):
    #    m_sf = ox.plot_graph_folium(G_sf, popup_attribute="name", weight=2, color="#8b0000", tiles='openstreetmap',
    #                                opacity=0.5)
    #    if x == 0:
    #        return ox.plot_route_folium(G_sf, routes_sf[0], route_map=m_sf, weight=5)
    #    else:
    #        return ox.plot_route_folium(G_sf, routes_sf[x], route_map=route_plot_sf(x - 1), weight=5)


    #map2 = route_plot_sf(50)

    st.write("Here, I plot the route between 50 elementary schools in San Francisco and the nearest bus station to them.")
    st.write("Obviously it takes a lot of time to run this code, so I show you already executed code.")
    st.write("For creating both maps I used OSMnx library and NetworkX - see the code for details.")

    filepathh = "map2.html"
    #map2.save(filepathh)

    HtmlFile1 = open(filepathh, 'r', encoding='utf-8')
    components.html(HtmlFile1.read(), height=500, width=700)


