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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Base temporary folder - uses system temp directory
BASE_UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_offset(reference_path, test_path, sr=22050, hop_length=512, threshold_ms=50):
    """
    Compute time offset (in ms) between reference and test audio.
    """
    try:
        ref, _ = librosa.load(reference_path, sr=sr, mono=True)
        test, _ = librosa.load(test_path, sr=sr, mono=True)
        
        if len(ref) == 0 or len(test) == 0:
            raise Exception("One of the audio files is empty or could not be loaded")
        
        # Trim leading/trailing silence
        ref, _ = librosa.effects.trim(ref)
        test, _ = librosa.effects.trim(test)
        
        # Compute RMS energy envelope
        ref_rms = librosa.feature.rms(y=ref, hop_length=hop_length)[0]
        test_rms = librosa.feature.rms(y=test, hop_length=hop_length)[0]
        
        # Normalize RMS to [0,1]
        ref_rms = (ref_rms - ref_rms.min()) / (ref_rms.max() - ref_rms.min() + 1e-10)
        test_rms = (test_rms - test_rms.min()) / (test_rms.max() - test_rms.min() + 1e-10)
        
        # Cross-correlate
        correlation = signal.correlate(test_rms, ref_rms, mode='same')
        lag = np.argmax(correlation) - len(ref_rms) // 2
        offset_ms = lag * hop_length / sr * 1000
        
        needs_review = abs(offset_ms) > threshold_ms
        return round(offset_ms, 2), needs_review
        
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Failed to process audio: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Create a unique sub-folder for THIS specific request
    request_id = uuid.uuid4().hex
    upload_path = os.path.join(BASE_UPLOAD_FOLDER, request_id)
    os.makedirs(upload_path, exist_ok=True)
    
    try:
        print(f"\n{'='*50}\n🔍 DEBUG: New request [{request_id}]\n{'='*50}")
        
        if 'reference' not in request.files or 'comparison[]' not in request.files:
            return jsonify({'error': 'Missing reference or comparison files'}), 400
        
        ref_file = request.files['reference']
        comp_files = [f for f in request.files.getlist('comparison[]') if f.filename != '']

        if ref_file.filename == '' or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Invalid reference file'}), 400
        
        if not comp_files:
            return jsonify({'error': 'No valid comparison files provided'}), 400

        # Save Reference
        ref_path = os.path.join(upload_path, f"ref_{ref_file.filename}")
        ref_file.save(ref_path)

        results = []
        for file in comp_files:
            if allowed_file(file.filename):
                temp_comp_path = os.path.join(upload_path, f"comp_{file.filename}")
                file.save(temp_comp_path)
                
                print(f"Processing: {file.filename}...")
                offset_ms, needs_review = compute_offset(ref_path, temp_comp_path)
                
                results.append({
                    'filename': file.filename,
                    'offset_ms': offset_ms,
                    'needs_review': needs_review
                })

        return jsonify({
            'reference': ref_file.filename,
            'results': results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # CLEANUP: Delete the entire folder for this request
        shutil.rmtree(upload_path, ignore_errors=True)
        print(f"🧹 Temporary files for {request_id} cleared.")

if __name__ == '__main__':
    # Network IP discovery for easy testing on other devices
    local_ip = os.popen('ipconfig getifaddr en0 2>/dev/null || echo "localhost"').read().strip()
    print(f"\n🚀 Server running at http://{local_ip}:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
