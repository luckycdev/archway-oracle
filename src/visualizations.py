import plotly.graph_objects as go
import pandas as pd

def build_traffic_chart(history_data, future_data, selected_date):
    """
    Creates a Plotly chart showing 24 hours of history and 12 hours of AI forecast.
    """
    fig = go.Figure()

    # 1. Add Historical Data (Actual Counts)
    fig.add_trace(go.Scatter(
        x=history_data['DateTime'],
        y=history_data['vehicle_count'],
        name='Historical Traffic',
        line=dict(color='#3366CC', width=3),
        fill='tozeroy',
        fillcolor='rgba(51, 102, 204, 0.1)'
    ))

    # 2. Add AI Forecast (Predicted Counts)
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'],
        y=future_data['Predicted_Vehicles'],
        name='AI Forecast',
        line=dict(color='#FF4B4B', width=4, dash='dot'),
        hovertemplate='%{y:.0f} vehicles/hr<br>%{x|%I:%M %p}'
    ))

    # 3. Add a Vertical Line for the "Present Moment" (WITHOUT annotation here)
    fig.add_vline(
        x=selected_date, 
        line_width=2, 
        line_dash="dash", 
        line_color="white"
    )

    # 4. Add the Annotation manually (This prevents the 'sum' error)
    fig.add_annotation(
        x=selected_date,
        y=1,
        yref="paper", # Anchors to the top of the chart
        text="Present Moment",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)",
        xanchor="right",
        xshift=-5
    )

    # 5. Styling the Layout
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=False,
            title="Time of Day",
            tickformat="%I %p"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Vehicles per Hour"
        ),
        hovermode="x unified"
    )

    return fig