import os
import uuid
import shutil
import tempfile
import numpy as np
import librosa
import traceback
from scipy import signal
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

MEDIA_VOLATILE_PATH = tempfile.gettempdir()
SUPPORTED_CONTAINERS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_CONTAINERS

def analyze_temporal_drift(anchor_path, rendition_path, sr=22050, hop_length=512, delta_threshold_ms=50):
    try:
        anchor_buffer, _ = librosa.load(anchor_path, sr=sr, mono=True)
        rendition_buffer, _ = librosa.load(rendition_path, sr=sr, mono=True)
        
        anchor_buffer, _ = librosa.effects.trim(anchor_buffer)
        rendition_buffer, _ = librosa.effects.trim(rendition_buffer)
        
        anchor_env = librosa.feature.rms(y=anchor_buffer, hop_length=hop_length)[0]
        rendition_env = librosa.feature.rms(y=rendition_buffer, hop_length=hop_length)[0]
        
        anchor_env = (anchor_env - anchor_env.min()) / (anchor_env.max() - anchor_env.min() + 1e-10)
        rendition_env = (rendition_env - rendition_env.min()) / (rendition_env.max() - rendition_env.min() + 1e-10)
        
        correlation = signal.correlate(rendition_env, anchor_env, mode='same')
        lag_frame = np.argmax(correlation) - len(anchor_env) // 2
        drift_ms = lag_frame * hop_length / sr * 1000
        
        # THE FIX: Cast to standard Python bool so JSON can handle it
        validation_flag = bool(abs(drift_ms) > delta_threshold_ms)
        return round(float(drift_ms), 2), validation_flag
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
                rendition_path = os.path.join(analysis_root, f"RENDITION_{track.filename}")
                track.save(rendition_path)
                drift, needs_val = analyze_temporal_drift(anchor_path, rendition_path)
                analysis_report.append({
                    'filename': track.filename,
                    'offset_ms': drift,
                    'needs_review': needs_val
                })

        return jsonify({'reference': anchor_track.filename, 'results': analysis_report})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(analysis_root, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
