import os
import uuid
import tempfile
import numpy as np
import librosa
import traceback
from scipy import signal
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Temporary folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_offset(reference_path, test_path, sr=22050, hop_length=512, threshold_ms=50):
    """
    Compute time offset (in ms) between reference and test audio.
    Positive offset means test is delayed relative to reference.
    Also returns a boolean indicating if manual review is needed (|offset| > threshold).
    """
    try:
        # Load audio files with error handling
        ref, _ = librosa.load(reference_path, sr=sr, mono=True)
        test, _ = librosa.load(test_path, sr=sr, mono=True)
        
        # Check if audio loaded properly
        if len(ref) == 0 or len(test) == 0:
            raise Exception("One of the audio files is empty or could not be loaded")
        
        # Trim leading/trailing silence to avoid false offsets from different padding
        ref, _ = librosa.effects.trim(ref)
        test, _ = librosa.effects.trim(test)
        
        # Compute RMS energy envelope
        ref_rms = librosa.feature.rms(y=ref, hop_length=hop_length)[0]
        test_rms = librosa.feature.rms(y=test, hop_length=hop_length)[0]
        
        # Normalize RMS to [0,1] to reduce amplitude influence
        ref_rms = (ref_rms - ref_rms.min()) / (ref_rms.max() - ref_rms.min() + 1e-10)
        test_rms = (test_rms - test_rms.min()) / (test_rms.max() - test_rms.min() + 1e-10)
        
        # Cross-correlate
        correlation = signal.correlate(test_rms, ref_rms, mode='same')
        lag = np.argmax(correlation) - len(ref_rms) // 2
        offset_ms = lag * hop_length / sr * 1000
        
        needs_review = abs(offset_ms) > threshold_ms
        return round(offset_ms, 2), needs_review
        
    except Exception as e:
        print(f"Error in compute_offset: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Failed to process audio: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Check that reference file is present
        if 'reference' not in request.files:
            return jsonify({'error': 'No reference file provided'}), 400
        ref_file = request.files['reference']

        # Check that comparison files are present
        if 'comparison[]' not in request.files:
            return jsonify({'error': 'No comparison files provided'}), 400
        comp_files = request.files.getlist('comparison[]')

        if ref_file.filename == '':
            return jsonify({'error': 'Reference file is empty'}), 400
        if len(comp_files) == 0:
            return jsonify({'error': 'Please upload at least one comparison file'}), 400

        # Validate reference file type
        if not allowed_file(ref_file.filename):
            return jsonify({'error': f'Reference file type not allowed: {ref_file.filename}'}), 400

        print(f"Processing reference file: {ref_file.filename}")
        
        # Save reference file with a unique name
        ref_ext = ref_file.filename.rsplit('.', 1)[1].lower()
        ref_unique = f"{uuid.uuid4().hex}.{ref_ext}"
        ref_path = os.path.join(UPLOAD_FOLDER, ref_unique)
        ref_file.save(ref_path)
        print(f"Saved reference to: {ref_path}")

        # Save comparison files, validating each
        saved_paths = [(ref_file.filename, ref_path)]  # reference first
        for idx, file in enumerate(comp_files):
            if file.filename == '':
                continue
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique = f"{uuid.uuid4().hex}.{ext}"
            path = os.path.join(UPLOAD_FOLDER, unique)
            file.save(path)
            saved_paths.append((file.filename, path))
            print(f"Saved comparison {idx+1}: {file.filename} -> {path}")

        # Compute offsets for each comparison file against the reference
        ref_filename, ref_path = saved_paths[0]
        results = []
        
        for filename, path in saved_paths[1:]:
            try:
                print(f"Processing {filename}...")
                offset_ms, needs_review = compute_offset(ref_path, path)
                results.append({
                    'filename': filename,
                    'offset_ms': offset_ms,
                    'needs_review': needs_review
                })
                print(f"✅ {filename}: offset={offset_ms}ms, needs_review={needs_review}")
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return jsonify({'error': error_msg}), 500

        # Clean up temporary files (optional - commented out for debugging)
        # for _, path in saved_paths:
        #     try:
        #         os.remove(path)
        #     except:
        #         pass

        return jsonify({
            'reference': ref_filename,
            'results': results
        })
        
    except Exception as e:
        print(f"Unexpected error in upload_files: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🎧 Audio Sync Checker Starting...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print("✅ Allowed formats: wav, mp3, m4a, flac, aac, mp4")
    print("🚀 Server running at http://127.0.0.1:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
