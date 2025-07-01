import asyncio
import os
import shutil
import uuid
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
from contextlib import asynccontextmanager
from .model import setup_transcription_pipeline
from .transcribe import transcribe_audio_stream

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "../templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# cache the pipeline before we start the app
async def preload_pipeline():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, setup_transcription_pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await preload_pipeline()
    yield
    # Clean up resources if needed


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    transcription_id = str(uuid.uuid4())

    async def stream_transcription():
        async for text_chunk in transcribe_audio_stream(file_path, transcription_id):
            yield f"data: {json.dumps({'text': text_chunk})}\n\n"
        yield f"data: {json.dumps({'text': '[STREAM ENDED]', 'end': True})}\n\n"
        # Clean up the uploaded file
        os.remove(file_path)

    return StreamingResponse(
        stream_transcription(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
