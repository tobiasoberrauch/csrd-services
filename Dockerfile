FROM python:3.11-slim as builder

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.5.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to export requirements - fix the command syntax
RUN poetry export --format requirements.txt --without-hashes --without-dev -o requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy requirements from builder stage
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create cache directory
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "csrd_services.main"]
