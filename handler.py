import runpod
import torch
import os
import boto3
import requests
import numpy as np
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from fish_speech.utils.inference import load_checkpoint, generate_tokens, decode_audio

model_manager = None

def init_model():
    global model_manager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "checkpoints/s1-mini"
    print(f"Loading Baked Model on {device}...")
    model_manager = load_checkpoint(model_path, device)
    print("✅ Model Ready and Offline!")

def upload_to_s3(file_path, file_name):
    """S3 Upload for long audio files"""
    s3 = boto3.client('s3', 
        aws_access_key_id=os.getenv('S3_KEY'), 
        aws_secret_access_key=os.getenv('S3_SECRET')
    )
    bucket = os.getenv('S3_BUCKET')
    s3.upload_file(file_path, bucket, file_name)
    return f"https://{bucket}.s3.amazonaws.com/{file_name}"

def handler(job):
    global model_manager
    if model_manager is None:
        init_model()

    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "").strip()
        ref_audio_url = job_input.get("ref_audio_url", "")
        
        if not text or not ref_audio_url:
            return {"status": "error", "message": "Required fields missing."}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Download & Process Reference
        ref_data = requests.get(ref_audio_url).content
        with open("ref.wav", "wb") as f: f.write(ref_data)
        
        # 2. Smart Chunking for 100,000 chars
        sentences = sent_tokenize(text)
        audio_segments = []
        
        print(f"Processing {len(sentences)} segments...")
        for i, s in enumerate(sentences):
            # Inference logic for fish-speech
            tokens = generate_tokens(model=model_manager.llama, text=s, device=device)
            wav_chunk = decode_audio(model_manager.dac, tokens)
            audio_segments.append(wav_chunk)
            
            # VRAM Garbage Collection (Crucial for long text)
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # 3. Join & Export (CPU Based for stability)
        final_wav = np.concatenate(audio_segments)
        int_audio = (final_wav * 32767).astype(np.int16)
        audio_out = AudioSegment(int_audio.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        
        output_filename = f"{job['id']}.mp3"
        audio_out.export(output_filename, format="mp3", bitrate="192k")
        
        # 4. Upload and Return URL
        s3_url = upload_to_s3(output_filename, output_filename)
        return {"status": "success", "s3_url": s3_url}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})