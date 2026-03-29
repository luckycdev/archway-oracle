import plotly.graph_objects as go

def build_traffic_chart(history_data, future_data, selected_date):
    fig = go.Figure()

    # 1. Past Actual Data (The history we already know)
    fig.add_trace(go.Scatter(
        x=history_data['DateTime'],
        y=history_data['vehicle_count'],
        name='Actual (Past)',
        line=dict(color='#3366CC', width=3),
        fill='tozeroy',
        fillcolor='rgba(51, 102, 204, 0.1)'
    ))

    # 2. Future Ground Truth (What ACTUALLY happens - for validation)
    # We plot this as a light grey dashed line
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'],
        y=future_data['vehicle_count'],
        name='Actual (Future Truth)',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=2, dash='dash'),
    ))

    # 3. AI Forecast (The model's prediction)
    fig.add_trace(go.Scatter(
        x=future_data['DateTime'],
        y=future_data['Predicted_Vehicles'],
        name='AI Forecast',
        line=dict(color='#FF4B4B', width=4, dash='dot'),
        hovertemplate='Predicted: %{y:.0f} cars<br>%{x|%I:%M %p}'
    ))

    # 4. Vertical "Present Moment" Line
    v_line_x = selected_date.strftime("%Y-%m-%d %H:%M:%S")
    fig.add_vline(x=v_line_x, line_width=2, line_dash="dash", line_color="white")
    
    fig.add_annotation(
        x=v_line_x, y=1, yref="paper", text="Current Time",
        showarrow=False, font=dict(color="white"), xanchor="right", xshift=-5
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Time of Day", tickformat="%I %p"),
        yaxis=dict(title="Vehicles per Hour"),
        hovermode="x unified"
    )

    return fig