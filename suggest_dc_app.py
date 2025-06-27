import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle, geodesic
import numpy as np

from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from math import radians

import geopandas as gpd
from shapely.geometry import Point

#------------------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Version 1.0.5 Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------


# Downloadable templates
st.markdown("### Download Template Files")
cust_template = pd.DataFrame(columns=["Customer_Code", "Lat", "Long", "Type", "Province"])
store_template = pd.DataFrame(columns=["Store_Code", "Lat", "Long", "Type", "Province"])
dc_template = pd.DataFrame(columns=["Hub_Name", "Lat", "Long", "Type", "Province"])

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("⬇️ Download Customer Template", cust_template.to_csv(index=False).encode('utf-8-sig'), "Customer_Template.csv", "text/csv")
with col2:
    st.download_button("⬇️ Download Store Template", store_template.to_csv(index=False).encode('utf-8-sig'), "Store_Template.csv", "text/csv")
with col3:
    st.download_button("⬇️ Download Hub Template", dc_template.to_csv(index=False).encode('utf-8-sig'), "Hub_Template.csv", "text/csv")

# Upload input files
cust_file = st.file_uploader("📤 Upload Customer File", type="csv")
store_file = st.file_uploader("📤 Upload Store File", type="csv")
dc_file = st.file_uploader("📤 Upload Hub File", type="csv")

# Load customer & store data
cust_data = store_data = None
if cust_file:
    try:
        cust_data = pd.read_csv(cust_file).dropna(subset=['Lat', 'Long'])
    except: st.stop()
if store_file:
    try:
        store_data = pd.read_csv(store_file).dropna(subset=['Lat', 'Long'])
    except: st.stop()

# Load maps
thailand = gpd.read_file("thailand.geojson")
thailand_union = thailand.unary_union
provinces_gdf = gpd.read_file("provinces.geojson")

# Filter inside Thailand
sources = []
if cust_data is not None:
    cust_data['geometry'] = cust_data.apply(lambda r: Point(r['Long'], r['Lat']), axis=1)
    gdf = gpd.GeoDataFrame(cust_data, geometry='geometry', crs="EPSG:4326")
    cust_data = gdf[gdf.geometry.within(thailand_union)].drop(columns='geometry')
    cust_data['Source'] = "Customer"
    cust_data = cust_data.rename(columns={"Customer_Code": "Code"})
    sources.append(cust_data)
if store_data is not None:
    store_data['geometry'] = store_data.apply(lambda r: Point(r['Long'], r['Lat']), axis=1)
    gdf = gpd.GeoDataFrame(store_data, geometry='geometry', crs="EPSG:4326")
    store_data = gdf[gdf.geometry.within(thailand_union)].drop(columns='geometry')
    store_data['Source'] = "Store"
    store_data = store_data.rename(columns={"Store_Code": "Code"})
    sources.append(store_data)

if not sources:
    st.warning("⚠️ Please upload at least one file: Customer or Store.")
    st.stop()

combined_data = pd.concat(sources, ignore_index=True)
all_types = combined_data['Type'].dropna().unique().tolist()
selected_types = st.multiselect("Filter Location Types:", options=all_types, default=all_types)
combined_data = combined_data[combined_data['Type'].isin(selected_types)]

# Nearest hub calculation
st.subheader("📍 Nearest Hub for Each Customer / Store")
if dc_file:
    dc_data = pd.read_csv(dc_file).dropna(subset=['Lat', 'Long'])
    dc_types = dc_data['Type'].dropna().unique().tolist()
    selected_dc_types = st.multiselect("Filter Hub Types:", options=dc_types, default=dc_types)
    dc_data = dc_data[dc_data['Type'].isin(selected_dc_types)]
    if 'Hub_Name' not in dc_data.columns:
        st.error("❌ 'Hub_Name' column is missing in hub data."); st.stop()

    # Fill missing province
    combined_data['Province'] = combined_data['Province'].fillna("Unknown")
    unknown = combined_data[combined_data['Province'].str.lower().isin(["", "unknown"])]
    if not unknown.empty:
        unknown['geometry'] = unknown.apply(lambda r: Point(r['Long'], r['Lat']), axis=1)
        gdf = gpd.GeoDataFrame(unknown, geometry='geometry', crs="EPSG:4326")
        joined = gpd.sjoin(gdf, provinces_gdf, how="left", predicate="within")
        joined['Province'] = joined['pro_en']
        known = combined_data[~combined_data.index.isin(unknown.index)]
        combined_data = pd.concat([known, joined[known.columns]], ignore_index=True)

    # Distance calculation
    combined_data[['Lat', 'Long']] = combined_data[['Lat', 'Long']].apply(pd.to_numeric, errors='coerce')
    combined_data = combined_data.dropna(subset=['Lat', 'Long'])

    cust_coords = np.radians(combined_data[['Lat', 'Long']])
    dc_coords = np.radians(dc_data[['Lat', 'Long']])
    hub_tree = BallTree(dc_coords, metric='haversine')
    distances, indices = hub_tree.query(cust_coords, k=1)
    distances_km = distances.flatten() * 6371

    combined_data['Nearest_Hub'] = dc_data.iloc[indices.flatten()]['Hub_Name'].values
    combined_data['Distance_km'] = np.round(distances_km, 2)

    st.dataframe(combined_data[['Code', 'Type', 'Province', 'Source', 'Nearest_Hub', 'Distance_km']])
    st.download_button("⬇️ Download Nearest Hub Results", combined_data.to_csv(index=False).encode('utf-8-sig'), "nearest_hub_results.csv", "text/csv")

        
    #------------------------------------------------------------------------------------------------------------------------
        
# Suggest New Hubs for Out-of-Radius Customers
st.subheader("Suggest New Hubs Based on Radius & Existing Hubs")
radius_threshold_km = st.slider("Set Radius Threshold from Existing Hubs (km):", 1, 500, 100)

# ------------------------------------------------------------------------------------------------------------------------

def kmeans_within_thailand(data, n_clusters, thailand_polygon, max_retry=10):
    # ✅ แนะนำ: simplify polygon เพื่อให้ within() เร็วขึ้น (ค่า 0.01 = ประมาณ 1 กม.)
    simplified_polygon = thailand_polygon.simplify(0.01)

    for i in range(max_retry):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42 + i)
        kmeans.fit(data[['Lat', 'Long']])
        centers = kmeans.cluster_centers_

        # ✅ แปลงศูนย์กลางเป็นจุด
        centers_geometry = gpd.GeoSeries(
            [Point(lon, lat) for lat, lon in centers],
            crs="EPSG:4326"
        )

        # ✅ คัดเฉพาะศูนย์กลางที่อยู่ในประเทศไทย
        valid_centers = [
            (lat, lon) for (lat, lon), point in zip(centers, centers_geometry)
            if point.within(simplified_polygon)
        ]

        if valid_centers:
            return valid_centers

    # ❗ fallback ถ้าไม่มี center ไหนเลยอยู่ในไทย
    return [(lat, lon) for lat, lon in centers]


# ------------------------ Main Block ------------------------

# -------------------- สร้าง BallTree และคำนวณระยะใกล้ที่สุด --------------------
hub_tree = BallTree(dc_coords, metric='haversine')
distances, _ = hub_tree.query(cust_coords, k=1)
distances_km = distances.flatten() * 6371  # คูณรัศมีโลก

# -------------------- ตัดสินว่าอยู่นอกระยะ hub เดิมหรือไม่ --------------------
combined_data['Outside_Hub'] = distances_km > radius_threshold_km

# -------------------- แปลงเป็น GeoDataFrame เพื่อตรวจสอบว่าอยู่ในประเทศไทย --------------------
combined_data['geometry'] = combined_data.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
combined_gdf = gpd.GeoDataFrame(combined_data, geometry='geometry', crs="EPSG:4326")
combined_gdf = combined_gdf[combined_gdf.geometry.within(thailand_union)]

# ✅ ลูกค้าที่อยู่ในประเทศไทยทั้งหมด
cluster_data = combined_gdf.copy()

st.markdown(
    f"<b>{len(cluster_data)} customers</b> will be used for new hub suggestions (in and out of coverage, inside Thailand).",
    unsafe_allow_html=True
)

if not cluster_data.empty:
    n_new_hubs = st.slider("How many new hubs to suggest from all customers?", 1, 30, 3)
    new_hub_locations = kmeans_within_thailand(cluster_data, n_new_hubs, thailand_union)

    st.subheader("New Hub Suggestions Map")
    m_new = folium.Map(location=[13.75, 100.5], zoom_start=6, control_scale=True)

    # ------------------------------------------------------------------------------------------------------------------------

    # Layer visibility controls
    with st.expander("🧭 Layer Visibility Controls"):
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        show_customer_markers = st.checkbox("Show Customer Markers", value=True)
        show_existing_hubs = st.checkbox("Show Existing Hubs", value=True)
        show_suggested_hubs = st.checkbox("Show Suggested Hubs", value=True)
        show_hub_radius_layer = st.checkbox("Show Existing Hub Radius Zones", value=True)

    #------------------------------------------------------------------------------------------------------------------------
    
        # Existing hub layer
        existing_layer = FeatureGroup(name="Existing Hubs")
        for _, row in dc_data.iterrows():
            folium.Marker(
                location=[row['Lat'], row['Long']],
                popup=f"Hub: {row['Hub_Name']}<br>Type: {row.get('Type', 'Unknown')}<br>Province: {row.get('Province', 'N/A')}",
                icon=folium.Icon(color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue', icon='store', prefix='fa')
            ).add_to(existing_layer)
        if show_existing_hubs:
            existing_layer.add_to(m_new)

        # Existing hub radius circles
        if show_hub_radius_layer:
            radius_layer = FeatureGroup(name="Existing Hub Radius")
            for _, row in dc_data.iterrows():
                folium.Circle(
                    location=[row['Lat'], row['Long']],
                    radius=radius_threshold_km * 1000,
                    color='gray',
                    fill=False,
                    dash_array="5"
                ).add_to(radius_layer)
            radius_layer.add_to(m_new)

        # Outside customer layer with brand-based color
        cust_gdf = combined_gdf[combined_gdf['Source'] == 'Customer'].copy()
        outside_customers = cust_gdf[cust_gdf['Outside_Hub'] == True]
        outside_layer = FeatureGroup(name="Outside Customers")
        for _, row in outside_customers.iterrows():
            color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue'
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.5,
                popup=row['Code']
            ).add_to(outside_layer)
        if show_customer_markers:
            outside_layer.add_to(m_new)

        # Suggested hub layer
        suggest_layer = FeatureGroup(name="Suggested New Hubs")
        
        for i, (lat, lon) in enumerate(new_hub_locations):
            point = Point(lon, lat)
        
            # หา province name
            province_name = "Unknown"
            for _, prov in provinces_gdf.iterrows():
                if point.within(prov['geometry']):
                    province_name = prov.get("pro_en", "Unknown")
                    break
        
            # แสดง marker + popup จังหวัด
            folium.Marker(
                location=[lat, lon],
                popup=f"Suggest New Hub #{i+1}<br>Province: {province_name}",
                icon=folium.Icon(color='darkgreen', icon='star', prefix='fa')
            ).add_to(suggest_layer)
        
            # วงรัศมี
            folium.Circle(
                location=[lat, lon],
                radius=radius_threshold_km * 1000,
                color='darkgreen',
                fill=True,
                fill_opacity=0.1,
                popup=f"Radius {radius_threshold_km} km"
            ).add_to(suggest_layer)
        
        # ✅ อย่าลืม add layer ถ้าเปิดให้แสดง
        if show_suggested_hubs:
            suggest_layer.add_to(m_new)
    
        # Combined heatmap
        if show_heatmap:
            heatmap_layer = FeatureGroup(name="Customer Heatmap")
            HeatMap(
                cust_data[['Lat', 'Long']].values.tolist(),
                radius=10,
                gradient={0.2: '#FFE5B4', 0.6: '#FFA500', 1: '#FF8C00'}
            ).add_to(heatmap_layer)
            heatmap_layer.add_to(m_new)

    LayerControl().add_to(m_new)

#------------------------------------------------------------------------------------------------------------------------
            
    st_folium(m_new, width=1100, height=600, key="new_hub_map", returned_objects=[], feature_group_to_add=None, center=[13.75, 100.5], zoom=6)
