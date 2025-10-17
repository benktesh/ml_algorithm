# ML Model Server

A Flask-based machine learning model server for customer churn prediction.

## Docker Setup

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier deployment)

### Building and Running with Docker

#### Option 1: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t ml-churn-model .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:80 ml-churn-model
   ```

#### Option 2: Using Docker Compose (Recommended)

1. **Start the application:**
   ```bash
   docker-compose up -d
   ```

2. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Testing the API

Once the container is running, you can test the API:

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Make a Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [5, 70.0, 350.0, 1, 0, 1]}'
```

### Docker Commands Reference

- **View running containers:** `docker ps`
- **View logs:** `docker logs <container_id>`
- **Stop container:** `docker stop <container_id>`
- **Remove container:** `docker rm <container_id>`
- **Remove image:** `docker rmi ml-churn-model`

### Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `PYTHONUNBUFFERED`: Ensures Python output is sent straight to terminal

### Security Features

- Non-root user execution
- Minimal base image (Python slim)
- Health check endpoint
- Proper dependency management

## Local Development

For local development without Docker:

1. **Create virtual environment:**
   ```bash
   python -m venv mlvenv
   source mlvenv/bin/activate  # Linux/Mac
   # or
   mlvenv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python serve_model.py
   ```