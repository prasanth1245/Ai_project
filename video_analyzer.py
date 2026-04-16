import cv2
import os
import csv
import numpy as np
import librosa
import pickle
import sys
from moviepy.editor import VideoFileClip
from deepface import DeepFace
from collections import Counter

# Set UTF-8 encoding for Windows emojis
sys.stdout.reconfigure(encoding='utf-8')

# --- 1. CONFIGURATION ---
VIDEO_PATH = "nakul.mp4" 
AUDIO_TEMP = "temp_audio_sync.wav"
REPORT_FILE = "analysis_report.csv"

# --- 2. LOAD TRAINED MODELS ---
try:
    with open('ser_model.pkl', 'rb') as f:
        ser_model = pickle.load(f)
    with open('ser_scaler.pkl', 'rb') as f:
        ser_scaler = pickle.load(f)
    print("SUCCESS: Models Loaded.")
except Exception as e:
    print(f"ERROR: .pkl files missing! {e}")
    exit()

RAVDESS_MAP = {'01':'Neutral', '02':'Calm', '03':'Happy', '04':'Sad', 
               '05':'Angry', '06':'Fearful', '07':'Disgust', '08':'Surprised'}

EMOTION_GROUPS = {
    'Happy': 'Positive', 'Surprised': 'Positive',
    'Sad': 'Negative', 'Angry': 'Negative', 'Fearful': 'Negative', 'Disgust': 'Negative',
    'Neutral': 'Neutral', 'Calm': 'Neutral'
}

# --- 3. PSYCHOLOGICAL GROUPING LOGIC ---
def get_emotion_group(emotion):
    return EMOTION_GROUPS.get(emotion, 'Neutral')

# --- 4. NORMALIZED AUDIO ENGINE ---
print("Extracting & Normalizing Audio Track...")
clip = VideoFileClip(VIDEO_PATH)
clip.audio.write_audiofile(AUDIO_TEMP, fps=48000, logger=None)
y, sr = librosa.load(AUDIO_TEMP, sr=48000)
y = librosa.util.normalize(y) # Fixed "Sad" bias

def analyze_vocal_tone(start_sec):
    start_sample = int(start_sec * sr)
    segment = y[start_sample : start_sample + int(2.5 * sr)]
    if len(segment) < 1000: return "Neutral"
    
    mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(segment)), sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr).T, axis=0)
    
    features = np.hstack([mfccs, chroma, mel])
    scaled = ser_scaler.transform(features.reshape(1, -1))
    return RAVDESS_MAP.get(ser_model.predict(scaled)[0], "Neutral")

# --- 5. PRECISE TEMPORAL ANALYSIS ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# FIX: Match Gradio's 1-second skip logic
skip_frames = int(fps) 

csv_file = open(REPORT_FILE, mode='w', encoding='utf-8', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Time', 'Face', 'Voice', 'Status'])

print(f"Starting Audit. Syncing with Gradio logic (Seeking every {skip_frames} frames).")



frame_count = 0
mismatch_count = 0 
total_audits = 0 

while cap.isOpened():
    # SYNC FIX: Teleport to the exact frame instead of sequential reading
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = cap.read()
    if not ret: break
    
    # Calculate timestamp exactly like the Gradio app
    current_sec = round(frame_count / fps, 2)
    
    # A. Face Analysis (Using OpenCV backend for parity)
    try:
        res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        f_emo = res[0]['dominant_emotion'].capitalize()
    except:
        f_emo = "Neutral"

    # B. Voice Analysis
    v_emo = analyze_vocal_tone(current_sec)
    
    # C. Group-Based Logic
    f_grp = get_emotion_group(f_emo)
    v_grp = get_emotion_group(v_emo)
    
    status = "Authentic"
    total_audits += 1
    if f_grp != v_grp and v_grp != "Neutral":
        status = "FAKE / MISMATCH"
        mismatch_count += 1
        
    writer.writerow([current_sec, f_emo, v_emo, status])
    print(f"[{current_sec}s] Face: {f_emo} | Voice: {v_emo} -> {status}")

    # Jump ahead
    frame_count += skip_frames

# --- 6. CLEANUP & FINAL VERDICT ---
cap.release()
csv_file.close()

if total_audits > 0:
    mismatch_rate = (mismatch_count / total_audits) * 100
    print(f"\n" + "="*40)
    print(f"            FINAL VERDICT")
    print(f"="*40)
    if mismatch_rate > 25:
        print(f"🚨 HIGH RISK ({mismatch_rate:.1f}% Inconsistency): Deception detected.")
    else:
        print(f"✅ AUTHENTIC ({100 - mismatch_rate:.1f}% Match): Expression consistent.")
    print(f"="*40)

if os.path.exists(AUDIO_TEMP):
    os.remove(AUDIO_TEMP)

print(f"Audit Complete. Report saved to {REPORT_FILE}")