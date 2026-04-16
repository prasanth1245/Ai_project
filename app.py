import gradio as gr
import cv2
import os
import pandas as pd
import numpy as np
import librosa
import pickle
from moviepy.editor import VideoFileClip
from deepface import DeepFace
from collections import Counter

# --- 1. SETUP & LOAD MODELS ---
def load_models():
    with open('ser_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('ser_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

ser_model, ser_scaler = load_models()
RAVDESS_MAP = {'01':'Neutral', '02':'Calm', '03':'Happy', '04':'Sad', 
               '05':'Angry', '06':'Fearful', '07':'Disgust', '08':'Surprised'}

EMOTION_GROUPS = {
    'Happy': 'Positive', 'Surprised': 'Positive',
    'Sad': 'Negative', 'Angry': 'Negative', 'Fearful': 'Negative', 'Disgust': 'Negative',
    'Neutral': 'Neutral', 'Calm': 'Neutral'
}

# --- 2. OPTIMIZED AUDIT FUNCTION ---
# --- UPDATED AUDIO EXTRACTION SECTION ---
# --- Change this section in your run_audit function ---
def run_audit(video_file, progress=gr.Progress()):
    if video_file is None:
        return "Please upload a video.", None, None

    AUDIO_TEMP = "gradio_audio_temp.wav"
    
    # Use 'with' to automatically close the file after use
    with VideoFileClip(video_file) as clip:
        clip.audio.write_audiofile(AUDIO_TEMP, fps=48000, logger=None)
    
    # Rest of your code...
    
    # Now that the 'with' block is done, MoviePy has released the file lock
    y, sr = librosa.load(AUDIO_TEMP, sr=48000)
    y = librosa.util.normalize(y) 
    
    # ... (rest of your cap = cv2.VideoCapture code)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
   # Changed from fps/2 to fps to match video_analyzer.pypython "app (1).py"
    skip_frames = int(fps) 
    
    audit_data = []
    mismatch_count = 0
    frame_count = 0
    
    # Progress Bar Initialization
    progress(0, desc="Starting Deception Audit...")

    while cap.isOpened():
        # Teleport to the next sample point
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret: break
        
        current_sec = round(frame_count / fps, 2)
        
        # A. Faster Facial Analysis (OpenCV Backend)
        try:
            res = DeepFace.analyze(frame, actions=['emotion'], 
                                   enforce_detection=False, 
                                   detector_backend='opencv')
            f_emo = res[0]['dominant_emotion'].capitalize()
        except:
            f_emo = "Neutral"

        # B. Vocal Analysis
        start_sample = int(current_sec * sr)
        segment = y[start_sample : start_sample + int(2.5 * sr)]
        v_emo = "Neutral"
        if len(segment) >= 1000:
            mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(segment)), sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr).T, axis=0)
            feat = np.hstack([mfccs, chroma, mel])
            scaled = ser_scaler.transform(feat.reshape(1, -1))
            v_emo = RAVDESS_MAP.get(ser_model.predict(scaled)[0], "Neutral")
        
        # C. Deception Logic
        f_grp = EMOTION_GROUPS.get(f_emo, "Neutral")
        v_grp = EMOTION_GROUPS.get(v_emo, "Neutral")
        status = "Authentic"
        if f_grp != v_grp and v_grp != "Neutral":
            status = "FAKE / MISMATCH"
            mismatch_count += 1
            
        audit_data.append([current_sec, f_emo, v_emo, status])
        
        # Update progress bar
        progress(frame_count / total_frames, desc=f"Analyzing {current_sec}s...")
        
        frame_count += skip_frames 

    cap.release()
    os.remove(AUDIO_TEMP)
    
    # --- 3. FIX: SAVE CSV TO DISK ---
    df = pd.DataFrame(audit_data, columns=['Time (s)', 'Face Emotion', 'Voice Emotion', 'Status'])
    report_filename = "emotion_audit_report.csv"
    df.to_csv(report_filename, index=False)
    
    # 4. Final Verdict
    mismatch_rate = (mismatch_count / len(df)) * 100 if len(df) > 0 else 0
    if mismatch_rate > 25:
        verdict = f"🚨 HIGH RISK ({mismatch_rate:.1f}% Inconsistency): Potential deception detected."
    else:
        verdict = f"✅ AUTHENTIC ({100 - mismatch_rate:.1f}% Match): Expressions are consistent."
        
    return verdict, df, report_filename

# --- 5. GRADIO UI DESIGN ---
def new_func():
    input_vid = gr.Video(label="Upload Subject Video")
    return input_vid

with gr.Blocks(title="AI Deception Auditor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Multimodal Emotion Authenticity Auditor")
    gr.Markdown("Detecting inconsistencies between facial expressions and vocal patterns.")
    
    with gr.Row():
        input_vid = new_func()
    
    with gr.Row():
        btn = gr.Button("🚀 Start Multi-Modal Audit", variant="primary")
    
    with gr.Row():
        out_verdict = gr.Textbox(label="Analysis Verdict")
    
    with gr.Row():
        out_table = gr.Dataframe(label="Detailed Audit Logs")
        out_file = gr.File(label="Download Full CSV Report")

    # Connect button to function
    btn.click(fn=run_audit, inputs=input_vid, outputs=[out_verdict, out_table, out_file])

if __name__ == "__main__":
    demo.launch()
