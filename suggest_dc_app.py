import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle

st.set_page_config(layout="wide")
st.title("📦 Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

# Downloadable template section
st.markdown("### 📥 Download Template Files")
cust_template = pd.DataFrame(columns=["Customer_Code", "Lat", "Long", "Type", "Province"])
dc_template = pd.DataFrame(columns=["DC_Name", "Lat", "Long", "Type", "Province"])

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download Customer Template",
        data=cust_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Customer_Template.csv',
        mime='text/csv'
    )
with col2:
    st.download_button(
        label="⬇️ Download DC Template",
        data=dc_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='DC_Template.csv',
        mime='text/csv'
    )

# Upload files
cust_file = st.file_uploader("Upload Customer File (.csv with Lat, Long, Customer_Code, Type, Province)", type="csv")
dc_file = st.file_uploader("Upload DC File (.csv with Lat, Long, DC_Name, Type, Province)", type="csv")

if cust_file:
    try:
        cust_data = pd.read_csv(cust_file)
        if cust_data.empty:
            st.error("🚫 Customer CSV file is empty. Please upload a file with data.")
            st.stop()
        cust_data = cust_data.dropna(subset=['Lat', 'Long'])
    except pd.errors.EmptyDataError:
        st.error("🚫 Customer CSV file is empty or corrupted.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load customer file: {e}")
        st.stop()

    locations = cust_data[['Lat', 'Long']]

    # Filter by Type
    customer_types = cust_data['Type'].dropna().unique().tolist()
    selected_types = st.multiselect("Filter Customer Types:", options=customer_types, default=customer_types)
    cust_data = cust_data[cust_data['Type'].isin(selected_types)]

    st.subheader("📍 Nearest DC for Each Customer")

    if dc_file:
        dc_data = pd.read_csv(dc_file).dropna(subset=['Lat', 'Long'])

        # Filter DC types
        dc_types = dc_data['Type'].dropna().unique().tolist()
        selected_dc_types = st.multiselect("Filter DC Types:", options=dc_types, default=dc_types)
        dc_data = dc_data[dc_data['Type'].isin(selected_dc_types)]

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
                'Type': row.get('Type', 'Unknown'),
                'Province': row.get('Province', 'N/A'),
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
    kmeans.fit(cust_data[['Lat', 'Long']])
    dc_locations = kmeans.cluster_centers_

    # Create folium map
    m = folium.Map(location=[13.75, 100.5], zoom_start=6)

    # Plot customer heatmap
    heat_data = cust_data[['Lat', 'Long']].values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)

    # Province-based Circle Visualization
    province_counts = cust_data['Province'].value_counts()
    for province, count in province_counts.items():
        subset = cust_data[cust_data['Province'] == province]
        if not subset.empty:
            lat, lon = subset[['Lat', 'Long']].mean()
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + count**0.5,
                color='blue',
                fill=True,
                fill_opacity=0.5,
                popup=f"{province}: {count} customers"
            ).add_to(m)

    # Clustered customer markers by Type
    customer_cluster = MarkerCluster(name="Customers")
    for _, row in cust_data.iterrows():
        type_lower = row.get('Type', '').lower()
        icon = 'home'
        color = 'lightblue' if type_lower == 'lotus' else 'red'
        popup_text = f"Customer: {row['Customer_Code']} ({row.get('Type', 'Unknown')})<br>Province: {row.get('Province', 'N/A')}"
        folium.Marker(
            [row['Lat'], row['Long']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(customer_cluster)
    m.add_child(customer_cluster)

    # Existing DCs by Type
    if dc_file:
        for _, row in dc_data.iterrows():
            type_lower = row.get('Type', '').lower()
            icon = 'store'
            color = 'lightblue' if type_lower == 'lotus' else 'red'
            popup_text = f"DC: {row['DC_Name']} ({row.get('Type', 'Unknown')})<br>Province: {row.get('Province', 'N/A')}"
            folium.Marker(
                [row['Lat'], row['Long']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)

    # Suggested DCs
    for i, (lat, lon) in enumerate(dc_locations):
        folium.Marker(
            location=[lat, lon],
            popup=f"Suggest Hub #{i+1}",
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
    st_folium(m, width=1100, height=600, returned_objects=[], key="main_map")
