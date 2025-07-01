# Transcriber

### Using Whisper running in local setup in [PyThorch](http://pytorch.org/)

Transcriber is a project designed to transcribe audio files into text using a Python backend. The project is organized into frontend and backend components, with the backend providing the core transcription logic and API endpoints.

## Features

- Audio file upload and transcription
- REST API for transcription
- Modular backend code structure
- Docker support for frontend deployment

## Project Structure

```
app.index.html           # Frontend entry point
backend/                 # Backend Python application
    pyproject.toml       # Python project configuration
    run_tests.sh         # Script to run backend tests
    start.sh             # Script to start backend server
    src/                 # Source code for backend
        app.py           # Main FastAPI app
        model.py         # Model loading and inference
        transcribe.py    # Transcription logic
        test_transcribe.py # Unit tests
    templates/           # HTML templates for backend
        index.html
infra/                   # Infrastructure files
    Dockerfile.frontend  # Dockerfile for frontend
```

## Getting Started

### Backend

1. Navigate to the `backend` directory:
   ```sh
   cd backend
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```sh
   uv pip install .
   ```
4. Start the backend server:
   ```sh
   ./start.sh
   ```

### Frontend

- Open `app.index.html` in your browser, or deploy using the provided Dockerfile in `infra/Dockerfile.frontend`.

### Running Tests

From the `backend` directory, run:

```sh
./run_tests.sh
```

## License

MIT License

## Author

Boris
