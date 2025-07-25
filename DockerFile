# Use official Python base image
FROM python:3.11-slim

# Set environment
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port (Cloud Run uses $PORT)
ENV PORT=8080
EXPOSE $PORT

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]