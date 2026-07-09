# 🎓 AI Proctoring System

## 📌 Overview

The **AI Proctoring System** is an intelligent online examination monitoring solution that uses Computer Vision and Deep Learning to detect suspicious activities during an online exam. The system continuously analyzes the candidate's behavior in real time and generates alerts when policy violations are detected. If the number of alerts exceeds the predefined threshold, the examination is automatically terminated.

This project aims to improve the integrity of online examinations by reducing the need for manual invigilation.

---

# ✨ Features

* 👁️ Eye Gaze Tracking
* 🙂 Head Pose Estimation
* 📦 Object Detection
* 🚨 Real-time Alert Generation
* ❌ Automatic Exam Termination after repeated violations
* 📹 Live Webcam Monitoring

---

# 🧠 Technologies Used

* Python
* OpenCV
* Dlib
* MediaPipe Face Mesh
* YOLOv4-Tiny
* Flask
* NumPy
* MySQL

---

# 🏗️ System Architecture

1. Capture live webcam video.
2. Detect the candidate's face.
3. Track eye movements using **Dlib**.
4. Estimate head orientation using **MediaPipe Face Mesh**.
5. Detect prohibited objects using **YOLOv4-Tiny**.
6. Monitor suspicious activities continuously.
7. Generate alerts for each violation.
8. Automatically terminate the examination if the alert count exceeds the allowed limit.

---

# 🔍 Core Features

## 1. Eye Gaze Tracking

**Technology:** Dlib Facial Landmark Detection

The system tracks the candidate's eye movements to determine whether they are looking at the screen or frequently looking away.

### Suspicious Activities Detected

* Looking left repeatedly
* Looking right repeatedly
* Looking up or down for extended periods
* Frequent gaze deviation from the screen

---

## 2. Head Pose Estimation

**Technology:** MediaPipe Face Mesh

The system estimates the candidate's head orientation using facial landmarks.

### Suspicious Activities Detected

* Turning the head left
* Turning the head right
* Looking away from the screen
* Continuous head movement

---

## 3. Object Detection

**Technology:** YOLOv4-Tiny

The object detection model identifies prohibited objects visible in the webcam feed.

### Examples of Detectable Objects

* Mobile phone
* Books
* Electronic devices
* Additional person (if supported by the trained model)

Whenever a prohibited object is detected, an alert is generated.

---

## 4. Alert Management

Every suspicious activity increases the alert count.

Examples include:

* Eye gaze violation
* Head pose violation
* Unauthorized object detection

### Exam Termination Rule

* The system continuously counts the alerts.
* If the candidate receives **more than 4 alerts**, the examination is **automatically terminated**.
* The termination helps maintain the fairness and integrity of the examination.

---

# 📊 Workflow

```text
Start Exam
      │
      ▼
Capture Webcam Frame
      │
      ▼
Face Detection
      │
      ├──────────────┐
      ▼              ▼
Eye Gaze        Head Pose
(Dlib)        (MediaPipe)
      │              │
      └──────┬───────┘
             ▼
      Object Detection
      (YOLOv4-Tiny)
             │
             ▼
   Suspicious Activity?
        │         │
       No        Yes
        │         │
        ▼         ▼
 Continue     Increase Alert Count
                  │
                  ▼
       Alert Count > 4 ?
            │        │
           No       Yes
            │        │
            ▼        ▼
       Continue   Terminate Exam
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Proctoring-System.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

# 💡 Future Enhancements

* Face recognition for candidate authentication
* Voice activity detection
* Multiple person detection
* Browser activity monitoring
* AI-based cheating risk score
* Cloud deployment
* Exam analytics dashboard

---

# 📚 Learning Outcomes

This project helped me gain practical experience in:

* Computer Vision
* Deep Learning
* Object Detection
* Facial Landmark Detection
* Head Pose Estimation
* Real-time Video Processing
* Flask Web Development
* AI-based Monitoring Systems

---

# 👩‍💻 Author

**Rehana Parveen S**

* GitHub: https://github.com/Rehanapar
* LinkedIn: https://linkedin.com/in/rehana-parveen-s-20388024b

---

