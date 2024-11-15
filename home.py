import pyaudio
import numpy as np
import time
import threading
import cv2
import re
import bcrypt
from gaze_tracking import GazeTracking
from utils.object_detection import load_yolo, load_classes, detect_objects
from db import get_db_connection
from config import Config
from flask import Flask, render_template, Response, redirect, url_for, jsonify, request
from flask_mysqldb import MySQL
import mediapipe as mp
 
app = Flask(__name__)
app.config.from_object(Config)
mysql = MySQL(app)
 
stop_camera_feed = False
alert_message = ""
noise_detected = False
 
# Audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024  # Increased chunk size for more data per frame
 
# Initialize PyAudio
p = pyaudio.PyAudio()
 
# Open the audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
 
 
def process_audio_chunk(chunk, sr):
    global noise_detected
    # Convert chunk to numpy array
    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max
 
    # Calculate energy (volume level)
    energy = np.sum(audio_data ** 2) / len(audio_data)
 
    # Lower threshold to detect low-level noise
    if energy > 0.0001:  # Adjust the threshold for better sensitivity
        noise_detected = True
    else:
        noise_detected = False
 
 
def start_audio_stream():
    global noise_detected
    while True:
        try:
            # Read audio chunk
            chunk = stream.read(CHUNK)
 
            # Process the audio chunk for noise detection
            process_audio_chunk(chunk, RATE)
 
            time.sleep(0.1)
        except IOError as e:
                print(f"Error reading audio stream: {e}")
 
 
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('name')
        password = request.form.get('password')
        if username and password:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT * FROM Information WHERE name = %s', (username,))
            account = cursor.fetchone()
            connection.close()
            if account and bcrypt.checkpw(password.encode('utf-8'), account['password'].encode('utf-8')):
                return redirect(url_for('instructions'))
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)
 
# Initialize variables for gaze tracking and head pose timing
looking_left_timer = 0
looking_right_timer = 0
looking_up_timer = 0
looking_down_timer = 0
gaze_left_timer = 0
gaze_right_timer = 0
gaze_up_timer = 0
gaze_down_timer = 0
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
# Threshold for alerts (in seconds)
ALERT_THRESHOLD = 4
TIME_THRESHOLD = 4
 
 
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')
 
@app.route('/instructions', methods=['GET', 'POST'])
def instructions():
    global stop_camera_feed
    if request.method == 'POST':
        stop_camera_feed = False
        return redirect(url_for('focus_alert'))
    return render_template('instructions.html')
 
@app.route('/focus_alert', methods=['GET', 'POST'])
def focus_alert():
    if request.method == 'POST':
        return redirect(url_for('index'))  # Redirect to index after clicking OK
    return render_template('focus_alert.html', alert_message="Please make sure you focus on the screen. Any deviation will be flagged as an alert.")
 
@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        if not all([username, password, email]):
            msg = 'Please fill out the form!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        else:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT * FROM Information WHERE name = %s', (username,))
            account = cursor.fetchone()
            if account:
                msg = 'Account already exists!'
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                cursor.execute('INSERT INTO Information (name, password, email) VALUES (%s, %s, %s)', (username, hashed_password.decode('utf-8'), email))
                connection.commit()
                msg = 'You have successfully registered!'
            connection.close()
    return render_template('reg.html', msg=msg)
 
# Initialize variables for gaze tracking and head pose timing
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
 
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/focus_feed')
def focus_feed():
    return Response(frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
def frame():
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FPS, 10)
 
    while True:
        # Read a frame from the webcam
        success, image = webcam.read()
        if not success:
            break
 
        # Encode the image as JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
 
        # Yield the frame as a part of the multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   
    webcam.release()
 
@app.route('/get_alert')
def get_alert():
    global alert_message
    return jsonify(alert=alert_message)
 
@app.route('/clear_alert', methods=['POST'])
def clear_alert():
    global alert_message
    alert_message = ""
    return jsonify(success=True)
 
@app.route('/too_many_alerts', methods=['GET', 'POST'])
def too_many_alerts():
    global stop_camera_feed
    if request.method == 'POST':
        stop_camera_feed = True
        return redirect(url_for('login'))
    return render_template('alert.html')
 
# The code for generating frames with the webcam and detecting gaze, head pose, etc.
def generate_frames():
    global alert_message, stop_camera_feed
    global looking_left_timer, looking_right_timer, looking_up_timer, looking_down_timer
    global gaze_left_timer, gaze_right_timer, gaze_up_timer, gaze_down_timer
 
    net, output_layers = load_yolo()
    classes = load_classes()
    phone_class_id = classes.index("cell phone")
    book_class_id = classes.index("book")
    person_class_id = classes.index("person")
 
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FPS, 10)
 
    while True:
        if stop_camera_feed:
            webcam.release()
            cv2.destroyAllWindows()
            break
 
        success, image = webcam.read()
        if not success:
            break
 
        image_h, image_w, _ = image.shape
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
 
        if results.multi_face_landmarks:
            face_2d, face_3d = [], []
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * image_w), int(lm.y * image_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
 
            focal_length = image_w
            cam_matrix = np.array([[focal_length, 0, image_w / 2], [0, focal_length, image_h / 2], [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)
 
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x_angle, y_angle = angles[0] * 360, angles[1] * 360
 
            if y_angle < -10:
                looking_left_timer += 1
                if looking_left_timer > TIME_THRESHOLD:
                    alert_message = "Alert: Looking left too long!"
            else:
                looking_left_timer = 0
 
            if y_angle > 10:
                looking_right_timer += 1
                if looking_right_timer > TIME_THRESHOLD:
                    alert_message = "Alert: Looking right too long!"
            else:
                looking_right_timer = 0
 
            if x_angle < -10:
                looking_down_timer += 1
                if looking_down_timer > TIME_THRESHOLD:
                    alert_message = "Alert: Looking down too long!"
            else:
                looking_down_timer = 0
 
            if x_angle > 10:
                looking_up_timer += 1
                if looking_up_timer > TIME_THRESHOLD:
                    alert_message = "Alert: Looking up too long!"
            else:
                looking_up_timer = 0
 
        gaze.refresh(image)
        gaze_text = "Unknown"
 
        if gaze.is_blinking():
            gaze_text = "Blinking"
            gaze_left_timer = gaze_right_timer = gaze_up_timer = gaze_down_timer = 0
        elif gaze.is_right():
            gaze_text = "Looking right"
            gaze_right_timer += 1
            if gaze_right_timer > ALERT_THRESHOLD:
                alert_message = "Alert: Gaze is looking left for too long!"
        elif gaze.is_left():
            gaze_text = "Looking left"
            gaze_left_timer += 1
            if gaze_left_timer > ALERT_THRESHOLD:
                alert_message = "Alert: Gaze is looking right for too long!"
        elif gaze.is_center():
            gaze_left_timer = gaze_right_timer = gaze_up_timer = gaze_down_timer = 0
        else:
            left_pupil_y = gaze.pupil_left_coords()[1] if gaze.pupil_left_coords() else None
            right_pupil_y = gaze.pupil_right_coords()[1] if gaze.pupil_right_coords() else None
            if left_pupil_y is not None and right_pupil_y is not None:
                if left_pupil_y < 0.3 * image_h and right_pupil_y < 0.3 * image_h:
                    gaze_text = "Looking up"
                    gaze_up_timer += 1
                    if gaze_up_timer > ALERT_THRESHOLD:
                        alert_message = "Alert: Gaze is looking up for too long!"
                elif left_pupil_y > 0.7 * image_h and right_pupil_y > 0.7 * image_h:
                    gaze_text = "Looking down"
                    gaze_down_timer += 1
                    if gaze_down_timer > ALERT_THRESHOLD:
                        alert_message = "Alert: Gaze is looking down for too long!"
                else:
                    gaze_up_timer = gaze_down_timer = 0
 
        boxes, confidences, class_ids = detect_objects(image, net, output_layers)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        phone_detected = False
        book_detected = False
        person_count = 0
 
        if len(indexes) > 0:
            for i in indexes.flatten():
                if class_ids[i] == phone_class_id:
                    phone_detected = True
                elif class_ids[i] == book_class_id:
                    book_detected = True
                elif class_ids[i] == person_class_id:
                    person_count += 1
 
        if phone_detected:
            alert_message = "Alert: Phone Detected!"
        elif book_detected:
            alert_message = "Alert: Book Detected!"
        elif person_count > 1:
            alert_message = "Alert: More than one Person Detected!"
        elif person_count == 0:
            alert_message = "Alert: No person detected!"
 
        # if noise_detected:
        #     alert_message = "Alert: Noise Detected!"
 
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
    webcam.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    audio_thread = threading.Thread(target=start_audio_stream, daemon=True)
    audio_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)