FROM python:3.11-slim

# Default Streamlit port
ARG SERVER_PORT=8501

# Allow docker-compose build args to override the default
ENV SERVER_PORT=${SERVER_PORT}
ENV PROCESSED_STREAM_PORT=8765

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Records which port the app listens on
EXPOSE 8501
EXPOSE 8765

# Run Streamlit on container startup
CMD ["sh", "-c", "python -m streamlit run src/app.py --server.port=${SERVER_PORT} --server.address=0.0.0.0"]