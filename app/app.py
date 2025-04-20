# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from transcriber import transcribe_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

latest_video_filename = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_video_filename
    video = request.files.get('video')
    if not video or video.filename == '':
        return jsonify({'error': 'No video provided'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(filepath)
    latest_video_filename = video.filename
    return jsonify({'video_url': f'/uploads/{video.filename}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/transcribe', methods=['GET'])
def transcribe():
    if not latest_video_filename:
        return jsonify({'transcription': 'No video uploaded.'}), 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_video_filename)
    transcription = transcribe_video(video_path)
    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    print("🔥 Silent Speech.AI Flask app running on http://127.0.0.1:5000")
    app.run(debug=True)
