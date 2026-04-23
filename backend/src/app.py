import asyncio
import os
import shutil
import uuid
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
from contextlib import asynccontextmanager
from .model import setup_transcription_pipeline
from .transcribe import (
    STREAM_END_MARKER,
    transcribe_audio_stream,
    kill_transcription,
)
from .llm_fixer import build_correction_chain, check_ollama_connection, MODEL_TYPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_READY = False


async def _load_model_background():
    global MODEL_READY
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, setup_transcription_pipeline)
    MODEL_READY = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_model_background())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health/model")
async def health_model():
    return {"ready": MODEL_READY}


@app.get("/api/health/ollama")
async def health_ollama():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, check_ollama_connection)
    result["model_type"] = MODEL_TYPE.value
    return result


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait.",
        )
    filename = file.filename or f"upload_{uuid.uuid4()}.bin"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    transcription_id = str(uuid.uuid4())
    chunks: list[str] = []
    corrected_chunks: list[str] = []

    async def fix_text_stream(text_to_fix: str):
        chain = build_correction_chain()
        yield "\nCorrected text:\n"
        async for chunk in chain(text_to_fix):
            yield chunk
            corrected_chunks.append(chunk)

    async def stream_transcription():
        try:
            async for text_chunk in transcribe_audio_stream(
                file_path, transcription_id
            ):
                if text_chunk != STREAM_END_MARKER:
                    chunks.append(text_chunk)
                    yield f"data: {json.dumps({'text': text_chunk, 'phase': 'transcribing'})}\n\n"
                else:
                    async for corrected_chunk in fix_text_stream("".join(chunks)):
                        yield f"data: {json.dumps({'text': corrected_chunk, 'phase': 'correcting'})}\n\n"
                    yield f"data: {json.dumps({'end': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'end': True})}\n\n"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return StreamingResponse(
        stream_transcription(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Transcription-Id": transcription_id,
        },
    )


@app.post("/kill_transcription/{transcription_id}")
async def kill_transcription_endpoint(transcription_id: str):
    try:
        kill_transcription(transcription_id)
        return {
            "status": "success",
            "message": f"Transcription {transcription_id} killed.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
