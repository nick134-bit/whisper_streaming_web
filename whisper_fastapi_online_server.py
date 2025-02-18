import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.whisper_streaming.whisper_online import backend_factory, online_factory, add_shared_args


import logging
import logging.config

def setup_logging():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(levelname)s [%(name)s]: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'root': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'loggers': {
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn.error': {
                'level': 'INFO',
            },
            'uvicorn.access': {
                'level': 'INFO',
            },
            'src.whisper_streaming.online_asr': {  
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'src.whisper_streaming.whisper_streaming': { 
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }

    logging.config.dictConfig(logging_config)

setup_logging()
logger = logging.getLogger(__name__)






app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add WebSocket configuration for lower latency
app.websocket_route_class = WebSocket
app.websocket_route_class.max_message_size = 1024 * 1024  # 1MB
app.websocket_route_class.timeout = 10.0  # 10 seconds timeout

parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

parser.add_argument(
    "--diarization",
    type=bool,
    default=False,
    help="Whether to enable speaker diarization.",
)


add_shared_args(parser)
args = parser.parse_args()

asr, tokenizer = backend_factory(args)

if args.diarization:
    from src.diarization.diarization_online import DiartDiarization



SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * 0.25  
BYTES_PER_SAMPLE = 2 
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE


async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process



@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    print("Loading online.")
    online = online_factory(args, asr, tokenizer)
    print("Online loaded.")

    if args.diarization:
        diarization = DiartDiarization(SAMPLE_RATE)

    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        
        chunk_history = []  # Will store dicts: {beg, end, text, speaker}
        
        while True:
            try:
                elapsed_time = int(time() - beg)
                beg = time()
                chunk = await loop.run_in_executor(
                    None, ffmpeg_process.stdout.read, 16000 * elapsed_time 
                )
                if (
                    not chunk
                ):  
                    chunk = await loop.run_in_executor(
                        None, ffmpeg_process.stdout.read, 2048 
                    )
                    if not chunk: 
                        print("FFmpeg stdout closed.")
                        break

                pcm_buffer.extend(chunk)

                if len(pcm_buffer) >= BYTES_PER_SEC * 0.25:  #
                    pcm_array = (
                        np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    pcm_buffer = bytearray()
                    online.insert_audio_chunk(pcm_array)
                    transcription = online.process_iter()
                    
                    if transcription:
                        chunk_history.append({
                        "beg": transcription.start,
                        "end": transcription.end,
                        "text": transcription.text,
                        "speaker": "0"
                        })
                    
                    full_transcription += transcription.text
                    if args.vac:
                        transcript = online.online.concatenate_tokens(online.online.transcript_buffer.buffer)
                    else:
                        transcript = online.concatenate_tokens(online.transcript_buffer.buffer)

                    buffer = transcript.text  
                    if buffer in full_transcription: # With VAC, the buffer is not updated until the next chunk is processed
                        buffer = ""
                                        
                    lines = [
                        {
                            "speaker": "0",
                            "text": "",
                        }
                    ]
                    
                    if args.diarization:
                        await diarization.diarize(pcm_array)
                        diarization.assign_speakers_to_chunks(chunk_history)

                    for ch in chunk_history:
                        if args.diarization and ch["speaker"] and ch["speaker"][-1] != lines[-1]["speaker"]:
                            lines.append(
                                {
                                    "speaker": ch["speaker"][-1],
                                    "text": ch['text'],
                                }
                            )
                        else:
                            lines[-1]["text"] += ch['text']

                    response = {"lines": lines, "buffer": buffer}
                    await websocket.send_json(response)
                    
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        del online
        
        if args.diarization:
            # Stop Diart
            diarization.close()




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=False,
        log_level="info"
    )
