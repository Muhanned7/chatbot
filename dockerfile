# Use official slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Copy requirements first (for caching)
COPY requirements.txt .

# Install CPU-only torch first (pre-built wheel, no compilation)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Now install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code + PDF
COPY . .

# Expose port
EXPOSE 8080

# Correct CMD: shell form with proper variable fallback and correct module name
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --worker-class sync --threads 8 --timeout 0 main:app