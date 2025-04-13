FROM python:3.11-slim as builder

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.5.1

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock* ./

# Install dependencies with Poetry
RUN poetry install --no-dev --only main

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy .venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY . .

# Set PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create cache directory
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "csrd_services.main"]