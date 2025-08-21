# Docker Setup for VocationVector

This guide explains how to run VocationVector using Docker, with the LLM model stored externally on your host machine.

## Prerequisites

- Docker and Docker Compose installed
- A GGUF format LLM model file (e.g., Qwen3-4B-Instruct-2507-F16.gguf)
- At least 8GB of RAM available for the LLM server

## Quick Start

1. **Set up environment variables**
   ```bash
   cp .env.docker.example .env
   ```
   
   Edit `.env` and set:
   - `LLM_MODEL_PATH`: Path to the directory containing your GGUF model
   - `LLM_MODEL_NAME`: Name of your GGUF model file

2. **Build and start the containers**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Web interface: http://localhost:5000
   - LLM server: http://localhost:8000

## Configuration

### Environment Variables

Key environment variables in `.env`:

- `LLM_MODEL_PATH`: Host directory containing the GGUF model file
- `LLM_MODEL_NAME`: Filename of the GGUF model
- `LLM_CTX_SIZE`: Context size for the LLM (default: 32768 for Qwen3-4B)
- `LLM_N_THREADS`: Number of threads for LLM inference (default: 4)
- `LLM_MEMORY_LIMIT`: Maximum memory for LLM container (default: 8G)
- `APP_ENVIRONMENT`: Application environment (development/production)

### Volume Mounts

The following directories are persisted as volumes:

- `./data/jobs`: Crawled job data
- `./data/resumes`: Resume files
- `./data/lancedb`: Vector database
- `./data/uploads`: User uploads
- `./data/pipeline_output`: Pipeline results
- `./data/.cache`: Cache data
- `./config`: Configuration files

The LLM model is mounted read-only from your host system.

## Common Commands

### Start services
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f llm-server
```

### Stop services
```bash
docker-compose down
```

### Rebuild after code changes
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Execute pipeline commands
```bash
# Full pipeline
docker-compose exec app python -m graph.pipeline --mode full_pipeline --query "python developer" --location "remote"

# Process resumes only
docker-compose exec app python -m graph.pipeline --mode process_resumes --resume-dir data/resumes

# View database stats
docker-compose exec app python -m graph.lancedb_cli stats
```

### Run tests
```bash
docker-compose exec app pytest tests/
```

## Troubleshooting

### LLM Server Issues

If the LLM server fails to start:

1. Check memory allocation:
   ```bash
   docker stats llm-server
   ```

2. Verify model file exists:
   ```bash
   docker-compose exec llm-server ls -la /models/
   ```

3. Check server logs:
   ```bash
   docker-compose logs llm-server
   ```

### Permission Issues

If you encounter permission errors with data directories:

```bash
# On Linux/Mac, ensure proper ownership
sudo chown -R $USER:$USER ./data
```

### Port Conflicts

If ports 5000 or 8000 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
services:
  app:
    ports:
      - "5001:5000"  # Change 5001 to your desired port
  
  llm-server:
    ports:
      - "8001:8000"  # Change 8001 to your desired port
```

## Production Deployment

For production deployment:

1. Use a `.env.production` file with appropriate settings
2. Consider using Docker Swarm or Kubernetes for orchestration
3. Set up proper logging and monitoring
4. Use a reverse proxy (nginx/traefik) for the web interface
5. Implement proper backup strategies for the data volumes

### Security Considerations

- The LLM model is mounted read-only to prevent modifications
- Ensure your `.env` file is not committed to version control
- Use Docker secrets for sensitive information in production
- Regularly update base images for security patches

## Resource Requirements

### Minimum Requirements
- CPU: 4 cores
- VRAM: 8GB+
- RAM: 16GB
- Storage: 10GB for application data

### Recommended Requirements
- CPU: 8+ cores
- VRAM: 16GB
- RAM: 32GB

## Monitoring

Monitor container health and resource usage:

```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Health check
docker-compose exec app curl http://localhost:5000/health
docker-compose exec llm-server curl http://localhost:8000/health
```
