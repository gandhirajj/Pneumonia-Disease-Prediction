# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy all source code and model files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run your Streamlit frontend
CMD ["streamlit", "run", "app1.py", "--server.port=8501", "--server.address=0.0.0.0"]
