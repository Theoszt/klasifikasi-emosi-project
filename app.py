from flask import Flask, request, render_template, redirect, url_for,send_from_directory
import joblib
import cv2
import numpy as np
import pywt
from PIL import Image
import io
from werkzeug.utils import secure_filename
import os
from sklearn.svm import SVC
app = Flask(__name__)

# Load model dan encoder (pastikan path benar)
pca = joblib.load(r'C:\project_smt 4\PCD\face emotion\pca_model.pkl')
label_encoder = joblib.load(r'C:\project_smt 4\PCD\face emotion\label_encoder.pkl')
svm_model = joblib.load(r'C:\project_smt 4\PCD\face emotion\svm_model.pkl')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_dwt_features(img_gray):
    coeffs = pywt.wavedec2(img_gray, wavelet='haar', level=3)
    features = []
    cA3 = coeffs[0]
    features.append(cA3.flatten())
    for detail_level in coeffs[1:]:
        for arr in detail_level:
            features.append(arr.flatten())
    feature_vector = np.hstack(features)
    return feature_vector

def crop_face_with_margin(gray_img, x, y, w, h, margin=0.1):
    height, width = gray_img.shape
    x_margin = int(w * margin)
    y_margin = int(h * margin)

    x1 = max(x - x_margin, 0)
    y1 = max(y - y_margin, 0)
    x2 = min(x + w + x_margin, width)
    y2 = min(y + h + y_margin, height)

    face_cropped = gray_img[y1:y2, x1:x2]
    return face_cropped

def predict_faces_in_frame(frame, margin=0.1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    results = []

    for (x, y, w, h) in faces:
        face_img = crop_face_with_margin(gray, x, y, w, h, margin=margin)
        face_resized = cv2.resize(face_img, (256, 256))
        face_normalized = face_resized.astype('float32') / 255.0

        features = extract_dwt_features(face_normalized)
        features_pca = pca.transform(features.reshape(1, -1))
        prediction_num = svm_model.predict(features_pca)
        prediction_proba = svm_model.predict_proba(features_pca)
        prediction_label = label_encoder.inverse_transform(prediction_num)

        results.append({
            'coords': (x, y, w, h),
            'label': prediction_label[0],
            'proba': prediction_proba[0]
        })

    return results

@app.route('/')
def home():
    return render_template(r'Home.html')

@app.route('/pelajari', methods=['GET'])
def pelajari():
    return render_template('Pelajari.html')
@app.route('/klasifikasi', methods=['GET'])
def klasifikasi_get():
    return render_template('klasifikasi.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'GET':
        return render_template('klasifikasi.html')

    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Membaca file upload langsung dari memory tanpa simpan ke disk
    img_bytes = file.read()
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_np = np.array(img_pil)
    # Konversi RGB (PIL) ke BGR (OpenCV)
    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Prediksi emosi wajah
    results = predict_faces_in_frame(frame)

    if not results:
        # Tidak terdeteksi wajah
        return render_template('klasifikasi.html', message="Wajah tidak terdeteksi. Silakan coba lagi.", results=None)

    for result in results:
        x, y, w, h = result['coords']
        label = result['label']
        proba = max(result['proba']) * 100
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({proba:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    import base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return render_template('klasifikasi.html', results=results, img_data=encoded_img)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    app.run(debug=True)
