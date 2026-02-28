import os
import uuid
import shutil
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Required for server-side rendering
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback
import acoustid # Requires: pip install pyacoustid
from scipy import signal
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# CONFIGURATION: Supporting high-res files and preventing 413 errors
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024

MEDIA_VOLATILE_PATH = tempfile.gettempdir()
SUPPORTED_CONTAINERS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_CONTAINERS

def generate_visual_comparison(anchor_y, rendition_y, drift_ms, match_score, sr):
    """Generates a base64 encoded waveform comparison image."""
    plt.figure(figsize=(10, 5), facecolor='#f8fafc')
    
    # Anchor Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(anchor_y, sr=sr, alpha=0.6, color='#3b82f6')
    plt.title(f"Sync: {drift_ms}ms | Content Integrity: {match_score}%", fontsize=10)
    plt.ylabel("Reference")
    plt.xticks([]) 
    
    # Rendition Waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(rendition_y, sr=sr, alpha=0.6, color='#f59e0b')
    plt.ylabel("Comparison")
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_temporal_drift(anchor_path, rendition_path, sr=22050, hop_length=512, delta_threshold_ms=50):
    try:
        # 1. Chromaprint Content Integrity Check
        _, fp_a = acoustid.fingerprint_file(anchor_path)
        _, fp_b = acoustid.fingerprint_file(rendition_path)
        match_score = round(acoustid.compare_fingerprints(fp_a, fp_b) * 100, 2)

        # 2. Signal Processing for Drift
        # Loading first 60s for speed and to avoid memory timeout
        anchor_buffer, _ = librosa.load(anchor_path, sr=sr, mono=True, duration=60)
        rendition_buffer, _ = librosa.load(rendition_path, sr=sr, mono=True, duration=60)
        
        a_trimmed, _ = librosa.effects.trim(anchor_buffer)
        r_trimmed, _ = librosa.effects.trim(rendition_buffer)
        
        # Envelope Cross-Correlation
        anchor_env = librosa.feature.rms(y=a_trimmed, hop_length=hop_length)[0]
        rendition_env = librosa.feature.rms(y=r_trimmed, hop_length=hop_length)[0]
        
        anchor_env = (anchor_env - anchor_env.min()) / (anchor_env.max() - anchor_env.min() + 1e-10)
        rendition_env = (rendition_env - rendition_env.min()) / (rendition_env.max() - rendition_env.min() + 1e-10)
        
        correlation = signal.correlate(rendition_env, anchor_env, mode='same')
        lag_frame = np.argmax(correlation) - len(anchor_env) // 2
        drift_ms = round(float(lag_frame * hop_length / sr * 1000), 2)
        
        # Logic: If content match is very low, the drift calculation is likely irrelevant
        validation_flag = bool(abs(drift_ms) > delta_threshold_ms or match_score < 40)
        
        # Generate visual for the first 15 seconds
        waveform_b64 = generate_visual_comparison(anchor_buffer[:sr*15], rendition_buffer[:sr*15], drift_ms, match_score, sr)
        
        return drift_ms, validation_flag, waveform_b64, match_score
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Signal processing failure: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"PROBE_{uuid.uuid4().hex[:8].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    
    try:
        if 'reference' not in request.files or 'comparison[]' not in request.files:
            return jsonify({'error': 'Missing files'}), 400
            
        anchor_track = request.files['reference']
        rendition_tracks = [f for f in request.files.getlist('comparison[]') if f.filename != '']

        anchor_path = os.path.join(analysis_root, f"ANCHOR_{anchor_track.filename}")
        anchor_track.save(anchor_path)

        analysis_report = []
        for track in rendition_tracks:
            if allowed_file(track.filename):
                r_path = os.path.join(analysis_root, f"RENDITION_{track.filename}")
                track.save(r_path)
                
                drift, needs_val, viz, score = analyze_temporal_drift(anchor_path, r_path)
                
                analysis_report.append({
                    'filename': track.filename,
                    'offset_ms': drift,
                    'match_confidence': score,
                    'needs_review': needs_val,
                    'visual': viz
                })

        return jsonify({'reference': anchor_track.filename, 'results': analysis_report})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(analysis_root, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
