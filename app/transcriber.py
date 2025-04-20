# transcriber.py
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown




# Google Drive file ID of your model
MODEL_ID = "17InB8yhTzhrRXt5xTUNH0Qwr2pszuiOB"
MODEL_PATH = "clean_lipnet_model.h5"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Vocabulary and conversion
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def preprocess_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        frame = tf.image.rgb_to_grayscale(frame)
        cropped = frame[190:236, 80:220, :]
        frames.append(cropped)

    cap.release()

    while len(frames) < 75:
        frames.append(tf.zeros_like(frames[0]))

    frames = frames[:75]
    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    normalized = tf.cast((frames - mean), tf.float32) / std

    return tf.expand_dims(normalized, axis=0).numpy()

def transcribe_video(video_path):
    try:
        input_tensor = preprocess_video(video_path)
        print("✅ Input shape:", input_tensor.shape)

        yhat = model.predict(input_tensor)
        print("✅ Model output shape:", yhat.shape)

        input_len = np.ones(yhat.shape[0]) * yhat.shape[1]
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=input_len, greedy=True)[0][0].numpy()
        print("✅ Decoded indices:", decoded)

        decoded_indices = decoded[0]
        chars = []
        for index in decoded_indices:
            if index != -1:
                char = num_to_char(index).numpy().decode("utf-8")
                chars.append(char)
        text = ''.join(chars)

        print("✅ Final text:", text)
        return text

    except Exception as e:
        return f"❌ Error: {str(e)}"