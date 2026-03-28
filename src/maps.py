import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

TRAFFIC_COLORS = {
    "🔴 Red (Heavy)":     "#FF0000",
    "🟡 Yellow (Moderate)": "#FFA500",
    "🟢 Green (Clear)":   "#00CC00",
}

def build_google_map(future_data, selected_segment, api_key):
    """
    Renders an interactive Google Map using DYNAMIC coordinates 
    from the TrafficTab23 dataset.
    """

    # 1. Calculate map center based on the data provided
    avg_lat = future_data['gps_latitude'].mean()
    avg_lng = future_data['gps_longitude'].mean()

    # 2. Build marker data dynamically from the unique road segments in the DF
    # We take the most recent prediction for each unique segment
    latest_preds = future_data.sort_values('DateTime').groupby('road_segment_id').last().reset_index()

    markers_js = ""
    for _, row in latest_preds.iterrows():
        segment_id = row['road_segment_id']
        lat = row['gps_latitude']
        lng = row['gps_longitude']
        traffic_level = row.get('Traffic_Level', '🟢 Green (Clear)')
        predicted_val = row.get('Predicted_Vehicles', 0)
        vehicles = int(predicted_val) if pd.notna(predicted_val) and not np.isnan(predicted_val) else 0
        color = TRAFFIC_COLORS.get(traffic_level, "#808080")

        # Highlight the selected segment
        scale = 12 if segment_id == selected_segment else 8
        
        # Add JavaScript for each marker
        markers_js += f"""
        {{
            const marker = new google.maps.Marker({{
                position: {{ lat: {lat}, lng: {lng} }},
                map: map,
                title: "Segment {segment_id}",
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: {scale},
                    fillColor: "{color}",
                    fillOpacity: 0.9,
                    strokeWeight: 2,
                    strokeColor: "#FFFFFF"
                }}
            }});

            const infoWindow = new google.maps.InfoWindow({{
                content: `
                    <div style="font-family: Arial; padding: 8px;">
                        <h3 style="margin:0; color:#333;">Road Segment: {segment_id}</h3>
                        <p style="margin:4px 0;"><b>Status:</b> {traffic_level}</p>
                        <p style="margin:4px 0;"><b>Predicted Flow:</b> {vehicles} vehicles/hr</p>
                    </div>
                `
            }});

            marker.addListener("click", () => {{
                infoWindow.open(map, marker);
            }});
        }}
        """

    # 3. Build the HTML (Center is now dynamic)
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #map {{ height: 480px; width: 100%; border-radius: 12px; }}
            body {{ margin: 0; padding: 0; background: transparent; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            function initMap() {{
                const map = new google.maps.Map(document.getElementById("map"), {{
                    zoom: 11,
                    center: {{ lat: {avg_lat}, lng: {avg_lng} }},
                    mapTypeId: "roadmap",
                    styles: [
                        {{ featureType: "poi", stylers: [{{ visibility: "off" }}] }},
                        {{ featureType: "transit", stylers: [{{ visibility: "off" }}] }}
                    ]
                }});
                {markers_js}
            }}
        </script>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
    </body>
    </html>
    """

    components.html(map_html, height=500)

def show_map_legend():
    st.markdown("""
    **Map Legend:**
    🔴 &nbsp; Heavy Traffic &nbsp;&nbsp;
    🟡 &nbsp; Moderate Traffic &nbsp;&nbsp;
    🟢 &nbsp; Clear Traffic &nbsp;&nbsp;
    ⚫ &nbsp; No Data
    """)