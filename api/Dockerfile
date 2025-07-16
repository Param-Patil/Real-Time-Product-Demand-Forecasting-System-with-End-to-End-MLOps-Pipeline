# Use slim Python image
FROM python:3.10-slim

# Install system dependencies (needed for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/api

# Copy and install dependencies
COPY requirements.txt /app/api/
RUN pip install --no-cache-dir -r requirements.txt

# Copy API app code and models
COPY ./api /app/api
COPY ./models /app/models

# Expose the port (Render uses 8000 by default)
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
