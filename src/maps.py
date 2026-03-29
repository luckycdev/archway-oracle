import streamlit as st
import streamlit.components.v1 as components

def show_map_legend():
    """Displays a small legend for the traffic colors."""
    st.markdown("""
        <div style="display: flex; gap: 10px; font-size: 0.8em; margin-bottom: 10px;">
            <span style="color: #28a745;">● Clear</span>
            <span style="color: #ffc107;">● Moderate</span>
            <span style="color: #dc3545;">● Heavy</span>
        </div>
    """, unsafe_allow_html=True)

def build_google_map(df, selected_segment, api_key):
    """
    Renders an interactive Google Map with St. Louis markers.
    """
    if df.empty:
        st.warning("No spatial data to display.")
        return

    # 1. Calculate the map center (Mean of St. Louis coordinates)
    center_lat = df['gps_latitude'].mean()
    center_lon = df['gps_longitude'].mean()

    # 2. Build the Marker Data for JavaScript
    # We convert the dataframe into a list of JS objects
    markers_js = ""
    for _, row in df.iterrows():
        color = "green"
        if "Red" in str(row.get('Traffic_Level', '')): color = "red"
        elif "Yellow" in str(row.get('Traffic_Level', '')): color = "yellow"
        
        # Highlight the selected segment with a bounce animation
        animation = "google.maps.Animation.BOUNCE" if row['road_segment_id'] == selected_segment else "null"
        
        markers_js += f"""
            new google.maps.Marker({{
                position: {{lat: {row['gps_latitude']}, lng: {row['gps_longitude']}}},
                map: map,
                title: "{row['road_segment_id']}",
                animation: {animation},
                icon: 'http://maps.google.com/mapfiles/ms/icons/{color}-dot.png'
            }});
        """

    # 3. Complete HTML/JS Template
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          #map {{ height: 100%; width: 100%; border-radius: 10px; }}
          html, body {{ height: 100%; margin: 0; padding: 0; }}
        </style>
      </head>
      <body>
        <div id="map"></div>
        <script>
          function initMap() {{
            const center = {{ lat: {center_lat}, lng: {center_lon} }};
            const map = new google.maps.Map(document.getElementById("map"), {{
              zoom: 12,
              center: center,
              styles: [
                {{ "elementType": "geometry", "stylers": [{{ "color": "#242f3e" }}] }},
                {{ "elementType": "labels.text.stroke", "stylers": [{{ "color": "#242f3e" }}] }},
                {{ "elementType": "labels.text.fill", "stylers": [{{ "color": "#746855" }}] }},
                {{ "featureType": "road", "elementType": "geometry", "stylers": [{{ "color": "#38414e" }}] }}
              ]
            }});
            {markers_js}
          }}
        </script>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
      </body>
    </html>
    """

    # 4. Render the component in Streamlit
    components.html(html_code, height=500)