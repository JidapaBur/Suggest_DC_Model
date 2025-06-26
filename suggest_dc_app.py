import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle

st.set_page_config(layout="wide")
st.title("📦 Customer & DC Visualization Tool")

# Upload files
cust_file = st.file_uploader("Upload Customer File (.csv with Lat, Long, Customer_Code)", type="csv")
dc_file = st.file_uploader("Upload DC File (.csv with Lat, Long, DC_Name)", type="csv")

if cust_file:
    cust_data = pd.read_csv(cust_file).dropna(subset=['Lat', 'Long'])
    locations = cust_data[['Lat', 'Long']]

    st.subheader("📍 Nearest DC for Each Customer")

    if dc_file:
        dc_data = pd.read_csv(dc_file).dropna(subset=['Lat', 'Long'])

        # Find nearest DC for each customer
        def find_nearest_dc(cust_lat, cust_lon, dc_df):
            distances = dc_df.apply(
                lambda row: great_circle((cust_lat, cust_lon), (row['Lat'], row['Long'])).km,
                axis=1
            )
            idx = distances.idxmin()
            return dc_df.loc[idx, 'DC_Name'], distances.min()

        results = []
        for _, row in cust_data.iterrows():
            nearest_dc, min_dist = find_nearest_dc(row['Lat'], row['Long'], dc_data)
            results.append({
                'Customer_Code': row['Customer_Code'],
                'Nearest_DC': nearest_dc,
                'Distance_km': round(min_dist, 2)
            })

        nearest_df = pd.DataFrame(results)
        st.dataframe(nearest_df)

    # Suggested DC clustering
    st.subheader("🌐 Suggested DC Locations (via KMeans Clustering)")
    n_dc = st.slider("Select number of suggested DCs:", 1, 10, 5)
    radius_km = st.slider("Radius per DC (km):", 10, 300, 100)
    radius_m = radius_km * 1000

    kmeans = KMeans(n_clusters=n_dc, random_state=42)
    kmeans.fit(locations)
    dc_locations = kmeans.cluster_centers_

    # Create folium map
    m = folium.Map(location=[13.75, 100.5], zoom_start=6)

    # Plot customer heatmap
    heat_data = cust_data[['Lat', 'Long']].values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)

    # Clustered customer markers
    customer_cluster = MarkerCluster(name="Customers")
    for _, row in cust_data.iterrows():
        folium.Marker(
            [row['Lat'], row['Long']],
            popup=f"Customer: {row['Customer_Code']}",
            icon=folium.Icon(color='blue', icon='user', prefix='fa')
        ).add_to(customer_cluster)
    m.add_child(customer_cluster)

    # Existing DCs
    if dc_file:
        for _, row in dc_data.iterrows():
            folium.Marker(
                [row['Lat'], row['Long']],
                popup=f"DC: {row['DC_Name']}",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(m)

    # Suggested DCs
    for i, (lat, lon) in enumerate(dc_locations):
        folium.Marker(
            location=[lat, lon],
            popup=f"Suggest DC #{i+1}",
            icon=folium.Icon(color='green', icon='star', prefix='fa')
        ).add_to(m)

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color='green',
            fill=True,
            fill_opacity=0.1,
            popup=f"Radius {radius_km} km"
        ).add_to(m)

    # Show final map
    st.subheader("🗺️ Visualization")
    st_data = st_folium(m, width=1100, height=600)
