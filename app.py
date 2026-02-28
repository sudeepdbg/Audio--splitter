import os
import uuid
import shutil
import hashlib
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback
import acoustid
import subprocess
from scipy import signal
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# CONFIGURATION
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024

# Create a 'data' folder for stable permissions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

SUPPORTED_CONTAINERS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

# In-memory cache for fingerprints to speed up multi-file analysis
FINGERPRINT_CACHE = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_CONTAINERS

def get_efficient_fingerprint(file_path):
    """Calculates or retrieves a cached fingerprint using a file hash."""
    # Hash the first 1MB of the file for a quick unique ID
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read(1024*1024)).hexdigest()
    
    if file_hash in FINGERPRINT_CACHE:
        return FINGERPRINT_CACHE[file_hash]
    
    # Generate fresh if not in cache using the verified shell path
    cmd = f"/opt/homebrew/bin/fpcalc -plain '{file_path}'"
    fp = subprocess.check_output(cmd, shell=True, timeout=30).decode().strip()
    
    FINGERPRINT_CACHE[file_hash] = fp
    return fp

def generate_visual_comparison(anchor_y, rendition_y, drift_ms, match_score, sr):
    plt.figure(figsize=(10, 5), facecolor='#f8fafc')
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(anchor_y, sr=sr, alpha=0.6, color='#3b82f6')
    plt.title(f"Sync: {drift_ms}ms | Content Integrity: {match_score}%", fontsize=10)
    plt.ylabel("Reference")
    plt.xticks([]) 
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(rendition_y, sr=sr, alpha=0.6, color='#f59e0b')
    plt.ylabel("Comparison")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_temporal_drift(anchor_path, rendition_path, sr=22050, hop_length=512):
    try:
        abs_anchor = os.path.abspath(anchor_path)
        abs_rendition = os.path.abspath(rendition_path)
        
        # 1. ENHANCED FINGERPRINTING & CACHING
        try:
            fp_a = get_efficient_fingerprint(abs_anchor)
            fp_b = get_efficient_fingerprint(abs_rendition)
            
            raw_match = acoustid.compare_fingerprints(fp_a, fp_b)
            match_score = round(raw_match * 100, 2)
            
            # Failsafe for identical files
            if fp_a == fp_b:
                match_score = 100.0
        except Exception as fp_err:
            print(f"DEBUG Fingerprint Error: {fp_err}")
            match_score = 0.0

        # 2. MEMORY-EFFICIENT LOADING (First 60s only)
        anchor_buffer, _ = librosa.load(abs_anchor, sr=sr, mono=True, duration=60)
        rendition_buffer, _ = librosa.load(abs_rendition, sr=sr, mono=True, duration=60)
        
        a_trimmed, _ = librosa.effects.trim(anchor_buffer)
        r_trimmed, _ = librosa.effects.trim(rendition_buffer)
        
        # 3. SIGNAL PROCESSING (RMS Envelopes)
        anchor_env = librosa.feature.rms(y=a_trimmed, hop_length=hop_length)[0]
        rendition_env = librosa.feature.rms(y=r_trimmed, hop_length=hop_length)[0]
        
        # Normalize
        anchor_env = (anchor_env - anchor_env.min()) / (anchor_env.max() - anchor_env.min() + 1e-10)
        rendition_env = (rendition_env - rendition_env.min()) / (rendition_env.max() - rendition_env.min() + 1e-10)
        
        # Cross-Correlation for Lag detection
        correlation = signal.correlate(rendition_env, anchor_env, mode='same')
        lag_frame = np.argmax(correlation) - len(anchor_env) // 2
        drift_ms = round(float(lag_frame * hop_length / sr * 1000), 2)
        
        # 4. NUANCED VALIDATION LOGIC
        issues = []
        if abs(drift_ms) > 100:
            issues.append("Severe desync (>100ms)")
        elif abs(drift_ms) > 50:
            issues.append("Minor desync (50-100ms)")
            
        if match_score < 30:
            issues.append("Content mismatch - wrong dub?")
        elif match_score < 60:
            issues.append("Low confidence match")
            
        validation_flag = len(issues) > 0
        
        viz = generate_visual_comparison(anchor_buffer[:sr*15], rendition_buffer[:sr*15], drift_ms, match_score, sr)
        
        return drift_ms, validation_flag, viz, match_score, issues

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Analysis failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        # Clear physical files
        for item in os.listdir(MEDIA_VOLATILE_PATH):
            item_path = os.path.join(MEDIA_VOLATILE_PATH, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        # Clear memory cache
        FINGERPRINT_CACHE.clear()
        return jsonify({'status': 'Cache and Memory cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    
    try:
        anchor_track = request.files['reference']
        rendition_tracks = request.files.getlist('comparison[]')

        anchor_path = os.path.join(analysis_root, anchor_track.filename)
        anchor_track.save(anchor_path)

        results = []
        for track in rendition_tracks:
            if track.filename and allowed_file(track.filename):
                r_path = os.path.join(analysis_root, track.filename)
                track.save(r_path)
                
                # Unpack the new return values including the 'issues' list
                drift, needs_val, viz, score, issues = analyze_temporal_drift(anchor_path, r_path)
                
                results.append({
                    'filename': track.filename, 
                    'offset_ms': drift,
                    'match_confidence': score, 
                    'needs_review': needs_val, 
                    'visual': viz,
                    'issues': issues
                })
        return jsonify({'reference': anchor_track.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
