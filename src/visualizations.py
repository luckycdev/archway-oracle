import plotly.graph_objects as go

COLOR_MAP = {
    "🔴 Red (Heavy)": "red",
    "🟡 Yellow (Moderate)": "yellow",
    "🟢 Green (Clear)": "green"
}

def build_traffic_chart(history_data, future_data, selected_date):
    """Build and return the main Plotly traffic forecast figure."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_data['DateTime'], y=history_data['Vehicles'],
        mode='lines', name='Known Past Traffic',
        line=dict(color='#1f77b4', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=future_data['DateTime'], y=future_data['Predicted_Vehicles'],
        mode='lines', name='AI Predicted Future',
        line=dict(color='#ff7f03', width=3, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=future_data['DateTime'], y=future_data['Vehicles'],
        mode='lines', name='What Actually Happened',
        line=dict(color='gray', width=1, dash='solid'), opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=future_data['DateTime'],
        y=future_data['Predicted_Vehicles'],
        mode='markers',
        name='Traffic Level',
        marker=dict(
            size=8,
            color=[COLOR_MAP[level] for level in future_data['Traffic_Level']]
        )
    ))

    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Number of Vehicles",
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