FROM python:3.8.20

COPY . /app

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Avoid numpy 2.0 issues (override any bad pip version early)
RUN pip install --upgrade pip==23.3.1

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Start the Flask app
CMD python app.py