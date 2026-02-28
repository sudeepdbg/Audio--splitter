import os
import uuid
import tempfile
import numpy as np
import librosa
from scipy import signal
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Temporary folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_offset(reference_path, test_path, sr=22050, hop_length=512, threshold_ms=50):
    """
    Compute time offset (in ms) between reference and test audio.
    Positive offset means test is delayed relative to reference.
    Also returns a boolean indicating if manual review is needed (|offset| > threshold).
    """
    ref, _ = librosa.load(reference_path, sr=sr, mono=True)
    test, _ = librosa.load(test_path, sr=sr, mono=True)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
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

    # Save reference file with a unique name
    ref_ext = ref_file.filename.rsplit('.', 1)[1].lower()
    ref_unique = f"{uuid.uuid4().hex}.{ref_ext}"
    ref_path = os.path.join(UPLOAD_FOLDER, ref_unique)
    ref_file.save(ref_path)

    # Save comparison files, validating each
    saved_paths = [(ref_file.filename, ref_path)]  # reference first
    for file in comp_files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(UPLOAD_FOLDER, unique)
        file.save(path)
        saved_paths.append((file.filename, path))

    # Compute offsets for each comparison file against the reference
    ref_filename, ref_path = saved_paths[0]
    results = []
    for filename, path in saved_paths[1:]:
        offset_ms, needs_review = compute_offset(ref_path, path)
        results.append({
            'filename': filename,
            'offset_ms': offset_ms,
            'needs_review': needs_review
        })

    # Optional: clean up temporary files after processing?
    # For a POC, you might keep them or implement a cleanup strategy.

    return jsonify({
        'reference': ref_filename,
        'results': results
    })

if __name__ == '__main__':
    # Use port 5001 to avoid AirPlay conflict on macOS
    app.run(debug=True, port=5001)
