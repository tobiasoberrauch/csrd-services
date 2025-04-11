# CSRD Services

This repository contains services for CSRD (Corporate Sustainability Reporting Directive) document classification and analysis.

## Installation

### Prerequisites
- Python 3.12+
- Poetry (dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/audius/csrd/csrd-services.git
cd /Users/tobiasoberrauch/Repositories/audius/csrd/csrd-services
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file with your HuggingFace token:
```bash
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```
Replace `your_token_here` with your actual token from [HuggingFace](https://huggingface.co/settings/tokens).

## Running the Application

### Development Mode
Run the application in development mode with hot-reloading:
```bash
poetry run dev
```

### Production Mode
Run the application in production mode:
```bash
poetry run start
```

By default, the server runs on `http://0.0.0.0:8010`.

## API Endpoints

### ESRS Classification

#### Upload and Classify a File
```
POST /api/v1/esrs-classification/upload
```
Upload a document (PDF, DOCX, PPTX, XLSX, TXT) for ESRS classification.

#### Classify Text
```
POST /api/v1/esrs-classification/text
```
Classify plain text input according to ESRS categories.

## Testing

Run the test suite:
```bash
poetry run tests
```

## Configuration

Configuration settings can be modified in the `.env` file or by setting environment variables:

- `SUPPORTED_FORMATS`: List of supported file formats
- `MAX_FILE_SIZE_MB`: Maximum file size in MB
- `SERVER_HOST`: Host address
- `SERVER_PORT`: Port number
- `LOG_LEVEL`: Logging level
- `HUGGINGFACE_TOKEN`: Authentication token for HuggingFace