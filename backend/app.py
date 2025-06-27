import asyncio
import time
import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
from transcribe import transcribe_audio_stream

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async def stream_transcription():
        async for text_chunk in transcribe_audio_stream(file_path):
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
