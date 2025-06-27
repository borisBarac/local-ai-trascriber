================================================
FILE: README.md
================================================

# Real Time Speech Transcription with FastRTC ⚡️and Local Whisper 🤗

This project uses FastRTC to handle the live audio streaming and open-source Automatic Speech Recognition models via Transformers.

Check the [FastRTC documentation](https://fastrtc.org/) to see what parameters you can tweak with respect to the audio stream, Voice Activity Detection (VAD), etc.

**System Requirements**

- python >= 3.10
- ffmpeg

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/sofi444/realtime-transcription-fastrtc
cd realtime-transcription-fastrtc
```

### Step 2: Set up environment

Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

### Step 3: Install ffmpeg

<details>
<summary>🍎 macOS</summary>

```bash
brew install ffmpeg
```

</details>

<details>
<summary>🐧 Linux (Ubuntu/Debian)</summary>

```bash
sudo apt update
sudo apt install ffmpeg
```

</details>

### Step 4: Configure environment

Create a `.env` file in the project root:

```env
UI_MODE=fastapi
APP_MODE=local
SERVER_NAME=localhost
```

- **UI_MODE**: controls the interface to use. If set to `gradio`, you will launch the app via Gradio and use their default UI. If set to anything else (eg. `fastapi`) it will use the `index.html` file in the root directory to create the UI (you can customise it as you want) (default `fastapi`).
- **APP_MODE**: ignore this if running only locally. If you're deploying eg. in Spaces, you need to configure a Turn Server. In that case, set it to `deployed`, follow the instructions [here](https://fastrtc.org/deployment/) (default `local`).
- **MODEL_ID**: HF model identifier for the ASR model you want to use (see [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)) (default `openai/whisper-large-v3-turbo`)
- **SERVER_NAME**: Host to bind to (default `localhost`)
- **PORT**: Port number (default `7860`)

### Step 5: Launch the application

```bash
python main.py
```

click on the url that pops up (eg. https://localhost:7860) to start using the app!

### Whisper

Choose the Whisper model version you want to use. See all [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending&search=whisper) - you can of course also use a non-Whisper ASR model.

On MPS, I can run `whisper-large-v3-turbo` without problems. This is my current favourite as it’s lightweight, performant and multi-lingual!

Adjust the parameters as you like, but remember that for real-time, we want the batch size to be 1 (i.e. start transcribing as soon as a chunk is available).

If you want to transcribe different languages, set the language parameter to the target language, otherwise Whisper defaults to translating to English (even if you set `transcribe` as the task).

================================================
FILE: index.html
================================================

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Whisper Transcription</title>
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #f9a45c 0%, #e66465 100%);
            --background-cream: #faf8f5;
            --text-dark: #2d2d2d;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-cream);
            color: var(--text-dark);
            min-height: 100vh;
        }

        .hero {
            background: var(--primary-gradient);
            color: white;
            padding: 2.5rem 2rem;
            text-align: center;
        }

        .hero h1 {
            font-size: 2.5rem;
            margin: 0;
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        .hero p {
            font-size: 1rem;
            margin-top: 0.5rem;
            opacity: 0.9;
        }

        .container {
            max-width: 1000px;
            margin: 1.5rem auto;
            padding: 0 2rem;
        }

        .transcript-container {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            padding: 1.5rem;
            height: 300px;
            overflow-y: auto;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .controls {
            text-align: center;
            margin: 1.5rem 0;
        }

        button {
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 0.95rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
            min-width: 180px;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(230, 100, 101, 0.15);
        }

        button:active {
            transform: translateY(0);
        }

        /* Transcript text styling */
        .transcript-container p {
            margin: 0.4rem 0;
            padding: 0.6rem;
            background: var(--background-cream);
            border-radius: 4px;
            line-height: 1.4;
            font-size: 0.95rem;
        }

        /* Custom scrollbar - made thinner */
        .transcript-container::-webkit-scrollbar {
            width: 6px;
        }

        .transcript-container::-webkit-scrollbar-track {
            background: var(--background-cream);
            border-radius: 3px;
        }

        .transcript-container::-webkit-scrollbar-thumb {
            background: #e66465;
            border-radius: 3px;
            opacity: 0.8;
        }

        .transcript-container::-webkit-scrollbar-thumb:hover {
            background: #f9a45c;
        }

        /* Add styles for toast notifications */
        .toast {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 16px 24px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 1000;
            display: none;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        .toast.error {
            background-color: #f44336;
            color: white;
        }

        .toast.warning {
            background-color: #ffd700;
            color: black;
        }

        /* Add styles for audio visualization */
        .icon-with-spinner {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            min-width: 180px;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid white;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            flex-shrink: 0;
        }

        .pulse-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            min-width: 180px;
        }

        .pulse-circle {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: white;
            opacity: 0.2;
            flex-shrink: 0;
            transform: translateX(-0%) scale(var(--audio-level, 1));
            transition: transform 0.1s ease;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
    </style>

</head>

<body>
    <!-- Add toast element after body opening tag -->
    <div id="error-toast" class="toast"></div>
    <div class="hero">
        <h1>Real-time Transcription</h1>
        <p>Powered by FastRTC and Local Whisper 🤗</p>
    </div>

    <div class="container">
        <div class="transcript-container" id="transcript"></div>
        <div class="controls">
            <button id="start-button">Start Recording</button>
        </div>
    </div>

    <script>
        let peerConnection;
        let webrtc_id;
        let audioContext, analyser, audioSource;
        let audioLevel = 0;
        let animationFrame;
        let eventSource;

        const startButton = document.getElementById('start-button');
        const transcriptDiv = document.getElementById('transcript');

        function showError(message) {
            const toast = document.getElementById('error-toast');
            toast.textContent = message;
            toast.style.display = 'block';

            // Hide toast after 5 seconds
            setTimeout(() => {
                toast.style.display = 'none';
            }, 5000);
        }

        function handleMessage(event) {
            // Handle any WebRTC data channel messages if needed
            const eventJson = JSON.parse(event.data);
            if (eventJson.type === "error") {
                showError(eventJson.message);
            }
            console.log('Received message:', event.data);
        }

        function updateButtonState() {
            if (peerConnection && (peerConnection.connectionState === 'connecting' || peerConnection.connectionState === 'new')) {
                startButton.innerHTML = `
                    <div class="icon-with-spinner">
                        <div class="spinner"></div>
                        <span>Connecting...</span>
                    </div>
                `;
            } else if (peerConnection && peerConnection.connectionState === 'connected') {
                startButton.innerHTML = `
                    <div class="pulse-container">
                        <div class="pulse-circle"></div>
                        <span>Stop Recording</span>
                    </div>
                `;
            } else {
                startButton.innerHTML = 'Start Recording';
            }
        }

        function setupAudioVisualization(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            audioSource = audioContext.createMediaStreamSource(stream);
            audioSource.connect(analyser);
            analyser.fftSize = 64;
            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            function updateAudioLevel() {
                analyser.getByteFrequencyData(dataArray);
                const average = Array.from(dataArray).reduce((a, b) => a + b, 0) / dataArray.length;
                audioLevel = average / 255;

                const pulseCircle = document.querySelector('.pulse-circle');
                if (pulseCircle) {
                    pulseCircle.style.setProperty('--audio-level', 1 + audioLevel);
                }

                animationFrame = requestAnimationFrame(updateAudioLevel);
            }
            updateAudioLevel();
        }

        async function setupWebRTC() {
            const config = __RTC_CONFIGURATION__;
            peerConnection = new RTCPeerConnection(config);

            const timeoutId = setTimeout(() => {
                const toast = document.getElementById('error-toast');
                toast.textContent = "Connection is taking longer than usual. Are you on a VPN?";
                toast.className = 'toast warning';
                toast.style.display = 'block';

                // Hide warning after 5 seconds
                setTimeout(() => {
                    toast.style.display = 'none';
                }, 5000);
            }, 5000);

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

                setupAudioVisualization(stream);

                stream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, stream);
                });

                // Add connection state change listener
                peerConnection.addEventListener('connectionstatechange', () => {
                    console.log('connectionstatechange', peerConnection.connectionState);
                    if (peerConnection.connectionState === 'connected') {
                        clearTimeout(timeoutId);
                        const toast = document.getElementById('error-toast');
                        toast.style.display = 'none';
                    }
                    updateButtonState();
                });

                // Create data channel for messages
                const dataChannel = peerConnection.createDataChannel('text');
                dataChannel.onmessage = handleMessage;

                // Create and send offer
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                await new Promise((resolve) => {
                    if (peerConnection.iceGatheringState === "complete") {
                        resolve();
                    } else {
                        const checkState = () => {
                            if (peerConnection.iceGatheringState === "complete") {
                                peerConnection.removeEventListener("icegatheringstatechange", checkState);
                                resolve();
                            }
                        };
                        peerConnection.addEventListener("icegatheringstatechange", checkState);
                    }
                });

                webrtc_id = Math.random().toString(36).substring(7);

                const response = await fetch('/webrtc/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type,
                        webrtc_id: webrtc_id
                    })
                });

                const serverResponse = await response.json();

                if (serverResponse.status === 'failed') {
                    showError(serverResponse.meta.error === 'concurrency_limit_reached'
                        ? `Too many connections. Maximum limit is ${serverResponse.meta.limit}`
                        : serverResponse.meta.error);
                    stop();
                    startButton.textContent = 'Start Recording';
                    return;
                }

                await peerConnection.setRemoteDescription(serverResponse);

                // Create event stream to receive transcripts
                eventSource = new EventSource('/transcript?webrtc_id=' + webrtc_id);
                eventSource.addEventListener("output", (event) => {
                    appendTranscript(event.data);
                });
            } catch (err) {
                clearTimeout(timeoutId);
                console.error('Error setting up WebRTC:', err);
                showError('Failed to establish connection. Please try again.');
                stop();
                startButton.textContent = 'Start Recording';
            }
        }

        function appendTranscript(text) {
            const p = document.createElement('p');
            p.textContent = text;
            transcriptDiv.appendChild(p);
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        }

        function stop() {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
                analyser = null;
                audioSource = null;
            }
            if (peerConnection) {
                if (peerConnection.getTransceivers) {
                    peerConnection.getTransceivers().forEach(transceiver => {
                        if (transceiver.stop) {
                            transceiver.stop();
                        }
                    });
                }

                if (peerConnection.getSenders) {
                    peerConnection.getSenders().forEach(sender => {
                        if (sender.track && sender.track.stop) sender.track.stop();
                    });
                }

                peerConnection.close();
                peerConnection = null;
            }
            // Close EventSource connection
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            audioLevel = 0;
            updateButtonState();
        }

        startButton.addEventListener('click', () => {
            if (startButton.textContent === 'Start Recording') {
                setupWebRTC();
            } else {
                stop();
            }
        });
    </script>

</body>

</html>

================================================
FILE: main.py
================================================
import os
import logging
import json

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastrtc import (
AdditionalOutputs,
ReplyOnPause,
Stream,
AlgoOptions,
SileroVadOptions,
audio_to_bytes,
)
from transformers import (
AutoModelForSpeechSeq2Seq,
AutoProcessor,
pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.logger_config import setup_logging
from utils.device import get_device, get_torch_and_np_dtypes
from utils.turn_server import get_rtc_credentials

load_dotenv()
setup_logging()
logger = logging.getLogger(**name**)

UI_MODE = os.getenv("UI_MODE", "fastapi")
APP_MODE = os.getenv("APP_MODE", "local")
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")

device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}")

attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
logger.info(f"Using attention: {attention}")

logger.info(f"Loading Whisper model: {MODEL_ID}")

try:
model = AutoModelForSpeechSeq2Seq.from_pretrained(
MODEL_ID,
torch_dtype=torch_dtype,
low_cpu_mem_usage=True,
use_safetensors=True,
attn_implementation=attention
)
model.to(device)
except Exception as e:
logger.error(f"Error loading ASR model: {e}")
logger.error(f"Are you providing a valid model ID? {MODEL_ID}")
raise

processor = AutoProcessor.from_pretrained(MODEL_ID)

transcribe_pipeline = pipeline(
task="automatic-speech-recognition",
model=model,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor,
torch_dtype=torch_dtype,
device=device,
)

# Warm up the model with empty audio

logger.info("Warming up Whisper model with dummy input")
warmup_audio = np.zeros((16000,), dtype=np_dtype) # 1s of silence
transcribe_pipeline(warmup_audio)
logger.info("Model warmup complete")

async def transcribe(audio: tuple[int, np.ndarray]):
sample_rate, audio_array = audio
logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")

    outputs = transcribe_pipeline(
        audio_to_bytes(audio),
        chunk_length_s=3,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    yield AdditionalOutputs(outputs["text"].strip())

logger.info("Initializing FastRTC stream")
stream = Stream(
handler=ReplyOnPause(
transcribe,
algo_options=AlgoOptions( # Duration in seconds of audio chunks (default 0.6)
audio_chunk_duration=0.6, # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
started_talking_threshold=0.2, # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
speech_threshold=0.1,
),
model_options=SileroVadOptions( # Threshold for what is considered speech (default 0.5)
threshold=0.5, # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
min_speech_duration_ms=250, # Max duration of speech chunks, longer will be split (default float('inf'))
max_speech_duration_s=30, # Wait for ms at the end of each speech chunk before separating it (default 2000)
min_silence_duration_ms=2000, # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
window_size_samples=1024, # Final speech chunks are padded by speech_pad_ms each side (default 400)
speech_pad_ms=400,
),
), # send-receive: bidirectional streaming (default) # send: client to server only # receive: server to client only
modality="audio",
mode="send",
additional_outputs=[
gr.Textbox(label="Transcript"),
],
additional_outputs_handler=lambda current, new: current + " " + new,
rtc_configuration=get_rtc_credentials(provider="hf") if APP_MODE == "deployed" else None
)

app = FastAPI()
stream.mount(app)

@app.get("/")
async def index():
html_content = open("index.html").read()
rtc_config = get_rtc_credentials(provider="hf") if APP_MODE == "deployed" else None
return HTMLResponse(content=html_content.replace("**RTC_CONFIGURATION**", json.dumps(rtc_config)))

@app.get("/transcript")
def \_(webrtc_id: str):
logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
async def output_stream():
try:
async for output in stream.output_stream(webrtc_id):
transcript = output.args[0]
logger.debug(f"Sending transcript for {webrtc_id}: {transcript[:50]}...")
yield f"event: output\ndata: {transcript}\n\n"
except Exception as e:
logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
raise

    return StreamingResponse(output_stream(), media_type="text/event-stream")

if **name** == "**main**":

    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)

    if UI_MODE == "gradio":
        logger.info("Launching Gradio UI")
        stream.ui.launch(
            server_port=port,
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        import uvicorn
        logger.info("Launching FastAPI server")
        uvicorn.run(app, host=server_name, port=port)

================================================
FILE: normal_gradio.py
================================================
import os
import logging

import gradio as gr
import numpy as np
from transformers import (
AutoModelForSpeechSeq2Seq,
AutoProcessor,
pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.logger_config import setup_logging
from utils.device import get_device, get_torch_and_np_dtypes

from dotenv import load_dotenv
load_dotenv()

setup_logging()
logger = logging.getLogger(**name**)

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")

device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(
f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}"
)

attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
logger.info(f"Using attention: {attention}")

logger.info(f"Loading Whisper model: {MODEL_ID}")

try:
model = AutoModelForSpeechSeq2Seq.from_pretrained(
MODEL_ID,
torch_dtype=torch_dtype,
low_cpu_mem_usage=True,
use_safetensors=True,
attn_implementation=attention
)
model.to(device)
except Exception as e:
logger.error(f"Error loading ASR model: {e}")
logger.error(f"Are you providing a valid model ID? {MODEL_ID}")
raise

processor = AutoProcessor.from_pretrained(MODEL_ID)

transcribe_pipeline = pipeline(
task="automatic-speech-recognition",
model=model,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor,
torch_dtype=torch_dtype,
device=device,
)

# Warm up the model with empty audio

logger.info("Warming up Whisper model with dummy input")
warmup_audio = np.zeros((16000,), dtype=np_dtype) # 1s of silence
transcribe_pipeline(warmup_audio)
logger.info("Model warmup complete")

async def transcribe(stream, audio: tuple[int, np.ndarray]):
sample_rate, audio_array = audio
logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")

    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    audio_array = audio_array.astype(np_dtype)
    audio_array /= np.max(np.abs(audio_array))

    if stream is not None:
        stream = np.concatenate((stream, audio_array))
    else:
        stream = audio_array

    outputs = transcribe_pipeline(
        {"sampling_rate": sample_rate, "raw": audio_array},
        chunk_length_s=10,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    return stream, outputs["text"].strip()

with gr.Blocks() as demo:
with gr.Row():
with gr.Column():
audio_input = gr.Audio(label="Audio Stream", streaming=True)
with gr.Column():
transcript = gr.Textbox(label="Transcript", value="")

        state = gr.State()
        audio_input.stream(
            transcribe,
            inputs=[state, audio_input],
            outputs=[state, transcript],
            stream_every=2
        )

        clear_button = gr.Button("Clear")
        clear_button.click(
            lambda: None, # clear the state
            outputs=[state]
        ).then(
            lambda: "", # clear the transcript
            outputs=[transcript]
        )

if **name** == "**main**":

    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)

    demo.launch(
        server_name=server_name,
        server_port=port,
        debug=True
    )

================================================
FILE: requirements.txt
================================================
accelerate==1.4.0
fastrtc==0.0.15
fastrtc[vad]==0.0.15
python-dotenv==1.0.1
transformers==4.49.0
torch==2.6.0
torchaudio==2.6.0

================================================
FILE: utils/**init**.py
================================================

================================================
FILE: utils/device.py
================================================
import torch
import numpy as np
import subprocess

def get_device(force_cpu=False):
if force_cpu:
return "cpu"
if torch.cuda.is_available():
return "cuda"
elif torch.backends.mps.is_available():
torch.mps.empty_cache()
return "mps"
else:
return "cpu"

def get_torch_and_np_dtypes(device, use_bfloat16=False):
if device == "cuda":
torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
np_dtype = np.float16
elif device == "mps":
torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
np_dtype = np.float16
else:
torch_dtype = torch.float32
np_dtype = np.float32
return torch_dtype, np_dtype

def cuda_version_check():
if torch.cuda.is_available():
try:
cuda_runtime = subprocess.check_output(["nvcc", "--version"]).decode()
cuda_version = cuda_runtime.split()[-2]
except Exception: # Fallback to PyTorch's built-in version if nvcc isn't available
cuda_version = torch.version.cuda

        device_name = torch.cuda.get_device_name(0)
        return cuda_version, device_name
    else:
        return None, None

================================================
FILE: utils/logger_config.py
================================================
import logging
import sys
import os

LOGS_DIR = "logs"

class ColorFormatter(logging.Formatter):
"""Custom formatter that adds colors to log levels"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    FORMATS = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging(level=None):
"""Configure logging for the entire application"""

    # Get level from environment variable or use default
    if level is None:
        level_name = os.getenv('LOG_LEVEL', 'INFO')
        level = getattr(logging, level_name.upper(), logging.INFO)

    # Configure stream handler (console output) with color formatter
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []
    root_logger.addHandler(stream_handler)

    # Prevent duplicate logging
    root_logger.propagate = False

    # Optionally configure file handler
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'app.log'))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

    # Get comma-separated list of loggers to suppress from env
    suppress_loggers = os.getenv('SUPPRESS_LOGGERS', '').strip()
    if suppress_loggers:
        for logger_name in suppress_loggers.split(','):
            logger_name = logger_name.strip()
            if logger_name:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info(f"Logging configured with level: {logging.getLevelName(level)}")

================================================
FILE: utils/turn_server.py
================================================
import os
from typing import Literal, Optional, Dict, Any
import requests

from fastrtc import get_hf_turn_credentials, get_twilio_turn_credentials

def get_rtc_credentials(
provider: Literal["hf", "twilio", "cloudflare"] = "hf",
\*\*kwargs
) -> Dict[str, Any]:
"""
Get RTC configuration for different TURN server providers.

    Args:
        provider: The TURN server provider to use ('hf', 'twilio', or 'cloudflare')
        **kwargs: Additional arguments passed to the specific provider's function

    Returns:
        Dictionary containing the RTC configuration
    """
    try:
        if provider == "hf":
            return get_hf_credentials(**kwargs)
        elif provider == "twilio":
            return get_twilio_credentials(**kwargs)
        elif provider == "cloudflare":
            return get_cloudflare_credentials(**kwargs)
    except Exception as e:
        raise Exception(f"Failed to get RTC credentials ({provider}): {str(e)}")

def get_hf_credentials(token: Optional[str] = None) -> Dict[str, Any]:
"""
Get credentials for Hugging Face's community TURN server.

    Required setup:
    1. Create a Hugging Face account at huggingface.co
    2. Visit: https://huggingface.co/spaces/fastrtc/turn-server-login
    3. Set HF_TOKEN environment variable or pass token directly
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    try:
        return get_hf_turn_credentials(token=token)
    except Exception as e:
        raise Exception(f"Failed to get HF TURN credentials: {str(e)}")

def get_twilio_credentials(
account_sid: Optional[str] = None,
auth_token: Optional[str] = None
) -> Dict[str, Any]:
"""
Get credentials for Twilio's TURN server.

    Required setup:
    1. Create a free Twilio account at: https://login.twilio.com/u/signup
    2. Get your Account SID and Auth Token from the Twilio Console
    3. Set environment variables:
       - TWILIO_ACCOUNT_SID (or pass directly)
       - TWILIO_AUTH_TOKEN (or pass directly)
    """
    account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        raise ValueError("Twilio credentials not found. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN env vars")

    try:
        return get_twilio_turn_credentials(account_sid=account_sid, auth_token=auth_token)
    except Exception as e:
        raise Exception(f"Failed to get Twilio TURN credentials: {str(e)}")

def get_cloudflare_credentials(
key_id: Optional[str] = None,
api_token: Optional[str] = None,
ttl: int = 86400
) -> Dict[str, Any]:
"""
Get credentials for Cloudflare's TURN server.

    Required setup:
    1. Create a free Cloudflare account
    2. Go to Cloudflare dashboard -> Calls section
    3. Create a TURN App and get the Turn Token ID and API Token
    4. Set environment variables:
       - TURN_KEY_ID
       - TURN_KEY_API_TOKEN

    Args:
        key_id: Cloudflare Turn Token ID (optional, will use env var if not provided)
        api_token: Cloudflare API Token (optional, will use env var if not provided)
        ttl: Time-to-live for credentials in seconds (default: 24 hours)
    """
    key_id = key_id or os.environ.get("TURN_KEY_ID")
    api_token = api_token or os.environ.get("TURN_KEY_API_TOKEN")

    if not key_id or not api_token:
        raise ValueError("Cloudflare credentials not found. Set TURN_KEY_ID and TURN_KEY_API_TOKEN env vars")

    response = requests.post(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json={"ttl": ttl},
    )

    if response.ok:
        return {"iceServers": [response.json()["iceServers"]]}
    else:
        raise Exception(
            f"Failed to get Cloudflare TURN credentials: {response.status_code} {response.text}"
        )

if **name** == "**main**": # Test
print(get_rtc_credentials(provider="hf"))
