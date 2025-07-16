# Use slim Python base
FROM python:3.10-slim

# System dependencies for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Set working directory
WORKDIR /app

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI code
COPY ./api ./api
COPY ./models ./models

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
