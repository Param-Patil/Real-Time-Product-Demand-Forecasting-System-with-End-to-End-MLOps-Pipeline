FROM python:3.10-slim

# System dependencies for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Set working directory inside container
WORKDIR /app/api

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire FastAPI app code including models
COPY . .

# Expose port (optional, but good practice)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
