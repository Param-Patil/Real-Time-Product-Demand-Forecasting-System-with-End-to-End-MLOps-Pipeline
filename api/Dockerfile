# Use official Python slim image
FROM python:3.10-slim

# Install LightGBM dependency
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the FastAPI app (main.py, models/)
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
