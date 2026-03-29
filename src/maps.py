import streamlit as st
import streamlit.components.v1 as components

def show_map_legend():
    """Displays a legend for the traffic colors and comparison marker."""
    st.markdown("""
        <div style="display: flex; gap: 15px; font-size: 0.85em; margin-bottom: 10px; font-weight: bold; flex-wrap: wrap;">
            <span style="color: #28a745;">● Clear</span>
            <span style="color: #ffc107;">● Moderate</span>
            <span style="color: #dc3545;">● Heavy</span>
            <span style="color: #007bff;">● Comparison Road</span>
        </div>
    """, unsafe_allow_html=True)

def build_google_map(df, selected_segment, api_key, secondary_segment=None):
    """
    Renders the Google Map with support for Primary and Secondary road highlighting.
    """
    if df.empty:
        st.warning("No spatial data to display.")
        return
      
    target_row = df[df['road_segment_id'] == selected_segment]
    if not target_row.empty:
        # Center the map EXACTLY on the selected street
        center_lat = target_row['gps_latitude'].iloc[0]
        center_lon = target_row['gps_longitude'].iloc[0]
        zoom_level = 15  # Zoomed in close!
    else:
        # Fallback: Center on St. Louis if something goes wrong
        center_lat = df['gps_latitude'].mean()
        center_lon = df['gps_longitude'].mean()
        zoom_level = {zoom_level}

    markers_js = ""
    for _, row in df.iterrows():
        color = "green"
        traffic_lvl = str(row.get('Traffic_Level', ''))
        if "Red" in traffic_lvl: color = "red"
        elif "Yellow" in traffic_lvl: color = "yellow"
        
        clean_name = str(row['road_segment_id']).replace("'", "\\'")
        is_primary = clean_name == str(selected_segment).replace("'", "\\'")
        is_secondary = secondary_segment and clean_name == str(secondary_segment).replace("'", "\\'")

        # Simplify: No bounce, just distinct icons
        icon_url = f'http://maps.google.com/mapfiles/ms/icons/{color}-dot.png'
        if is_secondary:
            icon_url = 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png'

        markers_js += f"""
            new google.maps.Marker({{
                position: {{lat: {row['gps_latitude']}, lng: {row['gps_longitude']}}},
                map: map,
                title: "{clean_name}",
                icon: '{icon_url}',
                zIndex: {(is_primary or is_secondary) and 1000 or 1}
            }});
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <style>#map {{ height: 100%; width: 100%; border-radius: 10px; }} html, body {{ height: 100%; margin: 0; }}</style>
      </head>
      <body>
        <div id="map"></div>
        <script>
          function initMap() {{
            const center = {{ lat: {center_lat}, lng: {center_lon} }};
            const map = new google.maps.Map(document.getElementById("map"), {{
              zoom: {zoom_level},
              center: center,
              styles: [
                {{ "elementType": "geometry", "stylers": [{{ "color": "#242f3e" }}] }},
                {{ "featureType": "road", "elementType": "geometry", "stylers": [{{ "color": "#38414e" }}] }},
                {{ "featureType": "road", "elementType": "labels.text.fill", "stylers": [{{ "color": "#9ca5b3" }}] }}
              ]
            }});
            {markers_js}
          }}
        </script>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
      </body>
    </html>
    """
    components.html(html_code, height=500)