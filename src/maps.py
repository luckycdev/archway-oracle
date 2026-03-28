import streamlit as st 
import streamlit.components.v1 as components

# Junction coordinates for St. Louis area
JUNCTION_COORDINATES = {
    1: {"lat": 38.6270, "lng": -90.1994, "name": "Junction 1 - Downtown"},
    2: {"lat": 38.6452, "lng": -90.2056, "name": "Junction 2 - Midtown"},
    3: {"lat": 38.6105, "lng": -90.1823, "name": "Junction 3 - South Grand / Southside"},
    4: {"lat": 38.6650, "lng": -90.2300, "name": "Junction 4 - North County / Northside"},
    5: {"lat": 38.6900, "lng": -90.1800, "name": "Junction 5 - Westport / Westside"},
    6: {"lat": 38.6000, "lng": -90.2100, "name": "Junction 6 - Lafayette Square / South Downtown"},
}

TRAFFIC_COLORS = {
    "🔴 Red (Heavy)":     "#FF0000",
    "🟡 Yellow (Moderate)": "#FFA500",
    "🟢 Green (Clear)":   "#00CC00",
}

def build_google_map(future_data, selected_junction, api_key):
    """
    Renders an interactive Google Map with traffic markers
    for each junction using the Maps JavaScript API.
    """

    # Build marker data from future predictions
    markers_js = ""
    for junction_id, coords in JUNCTION_COORDINATES.items():
        junction_future = future_data[future_data['Junction'] == junction_id]

        if not junction_future.empty:
            next_pred = junction_future.iloc[0]
            traffic_level = next_pred['Traffic_Level']
            vehicles = int(next_pred['Predicted_Vehicles'])
            color = TRAFFIC_COLORS.get(traffic_level, "#808080")
        else:
            traffic_level = "No Data"
            vehicles = 0
            color = "#808080"

        # Highlight the selected junction with a larger marker
        scale = 12 if junction_id == selected_junction else 8

        markers_js += f"""
        {{
            const marker = new google.maps.Marker({{
                position: {{ lat: {coords['lat']}, lng: {coords['lng']} }},
                map: map,
                title: "{coords['name']}",
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
                        <h3 style="margin:0; color:#333;">{coords['name']}</h3>
                        <p style="margin:4px 0;"><b>Status:</b> {traffic_level}</p>
                        <p style="margin:4px 0;"><b>Predicted Vehicles:</b> {vehicles}/hr</p>
                    </div>
                `
            }});

            marker.addListener("click", () => {{
                infoWindow.open(map, marker);
            }});
        }}
        """

    # Full Google Maps HTML component
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #map {{
                height: 480px;
                width: 100%;
                border-radius: 12px;
            }}
            body {{ margin: 0; padding: 0; background: transparent; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            function initMap() {{
                const map = new google.maps.Map(document.getElementById("map"), {{
                    zoom: 12,
                    center: {{ lat: 38.6274, lng: -90.1982 }},
                    mapTypeId: "roadmap",
                    styles: [
                        {{ featureType: "poi", stylers: [{{ visibility: "off" }}] }},
                        {{ featureType: "transit", stylers: [{{ visibility: "off" }}] }}
                    ]
                }});

                {markers_js}
            }}
        </script>
        <script
            src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap"
            async defer>
        </script>
    </body>
    </html>
    """

    components.html(map_html, height=500)


def show_map_legend():
    """Renders a simple color legend for the map."""
    st.markdown("""
    **Map Legend:**
    🔴 &nbsp; Heavy Traffic &nbsp;&nbsp;
    🟡 &nbsp; Moderate Traffic &nbsp;&nbsp;
    🟢 &nbsp; Clear Traffic &nbsp;&nbsp;
    ⚫ &nbsp; No Data
    """)