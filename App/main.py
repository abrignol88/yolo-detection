from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import os, io
import numpy as np

from yolo_predictions import YOLO_Pred
import cv2
from PIL import Image
import base64


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/yolo_image', methods=['GET','POST'])
def yolo_image():
    if request.method == 'GET':
        return render_template('yolo_image.html')
    
    elif request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_url = url_for('get_file', filename=filename)
      
            # Object detection
            img = Image.open(file_path).convert("RGB")
            img = np.array(img)
            pred_img = yolo.predictions(img)
            pred_img_obj = Image.fromarray(pred_img)

            # Save image into buffer
            data = io.BytesIO()
            pred_img_obj.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
        
            return render_template('yolo_image_detection.html', file_url=file_url, img_data=encoded_img_data.decode('utf-8'))


def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if frame is None:
            break
        frame = yolo.predictions(np.array(frame))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/yolo_video')
def yolo_video():
    return render_template('yolo_video.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
    