FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Copy the rest of the application
COPY . .

# Create cache directory
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "start"]