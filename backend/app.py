import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stream")
async def stream_text():
    async def generate_text():
        sample_texts = [
            "Hello, this is streaming text!",
            "New message every 0.5 seconds...",
            "Real-time updates working perfectly!",
            "FastAPI streaming is awesome!",
            "Server-Sent Events in action!",
            "More text coming your way...",
            "This is message number 7",
            "Keep watching for more updates!",
            "Almost there with the demo...",
            "Final streaming message!",
        ]

        for i, text in enumerate(sample_texts):
            # Format as Server-Sent Events
            timestamp = time.strftime("%H:%M:%S")
            message = f"[{timestamp}] {text}"

            # SSE format: data: {content}\n\n
            yield f"data: {json.dumps({'text': message, 'id': i})}\n\n"

            # Wait 0.5 seconds before next message
            await asyncio.sleep(0.5)

        # Send end signal
        yield f"data: {json.dumps({'text': '[STREAM ENDED]', 'end': True})}\n\n"

    return StreamingResponse(
        generate_text(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
