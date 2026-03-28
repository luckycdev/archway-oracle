import plotly.graph_objects as go
import numpy as np

COLOR_MAP = {
    "🔴 Red (Heavy)": "red",
    "🟡 Yellow (Moderate)": "yellow",
    "🟢 Green (Clear)": "green"
}

def build_traffic_chart(history_data, future_data, selected_date):
    """Build and return the main Plotly traffic forecast figure."""
    fig = go.Figure()

    # 1. Known Past
    fig.add_trace(go.Scatter(
        x=history_data['DateTime'], y=history_data['vehicle_count'],
        mode='lines', name='Historical Flow',
        line=dict(color='#1f77b4', width=3)
    ))

    # 2. AI Forecast
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'], y=future_data['Predicted_Vehicles'],
        mode='lines', name='AI Predicted Future',
        line=dict(color='#ff7f03', width=3, dash='dot')
    ))

    # 3. Ground Truth (What actually happened)
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'], y=future_data['vehicle_count'],
        mode='lines', name='Actual Flow',
        line=dict(color='gray', width=1, dash='solid'), opacity=0.5
    ))

    # 4. Traffic Level Markers
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'],
        y=future_data['Predicted_Vehicles'],
        mode='markers',
        name='Traffic Status',
        marker=dict(
            size=10,
            color=[COLOR_MAP.get(level, "gray") for level in future_data['Traffic_Level']],
            line=dict(width=1, color='white') 
        ),
        customdata=np.stack((future_data['road_segment_id'], future_data['gps_latitude'], future_data['gps_longitude']), axis=-1),
        hovertemplate=(
            "<b>Road Segment:</b> %{customdata[0]}<br>" +
            "<b>Location:</b> %{customdata[1]:.4f}, %{customdata[2]:.4f}<br>" +
            "<b>Predicted Count:</b> %{y}<extra></extra>"
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Timeline",
        yaxis_title="Vehicle Count",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        shapes=[dict(
            type="line",
            x0=selected_date, x1=selected_date,
            y0=0, y1=1, yref="paper",
            line=dict(color="White", width=2, dash="dash")
        )],
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig