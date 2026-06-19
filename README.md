# Smart Attendance System using Face Recognition and Uniform Detection

## 📌 Project Overview

The **Smart Attendance System** is a Python-based desktop application that automates student attendance using **Face Recognition** and **Uniform Color Detection**. The application captures student faces, trains a recognition model, identifies students through a webcam, verifies whether they are wearing the required uniform, and records attendance with both **In-Time** and **Out-Time**.

The project provides a simple graphical user interface (GUI) built with **Tkinter**, making it easy to register students, train the model, recognize faces, and view attendance records.

---

## ✨ Features

* Student Registration using ID and Name
* Face Image Capture (100 images per student)
* Face Recognition using LBPH Algorithm
* Automatic Attendance Marking
* In-Time and Out-Time Recording
* Uniform Detection (Blue Shirt + Red Tie)
* Attendance stored in CSV format
* User-friendly Tkinter GUI
* Attendance Viewer

---

## 🛠 Technologies Used

* Python 3.x
* Tkinter
* OpenCV
* NumPy
* Pandas
* Pillow (PIL)
* CSV
* DateTime

---

## 📂 Project Structure

```
Smart-Attendance-System/
│
├── Attendance/
│   └── attendance.csv
│
├── TrainingImage/
│
├── TrainingImageLabel/
│   └── Trainner.yml
│
├── UserDetails/
│   └── UserDetails.csv
│
├── haarcascade_frontalface_default.xml
│
├── main.py
│
└── README.md
```

---

## ⚙️ Working Process

### 1. Add Student

* Enter Student ID and Name.
* Capture 100 face images using the webcam.
* Save images in the `TrainingImage` folder.
* Store student details in `UserDetails.csv`.

### 2. Train Model

* Read all captured face images.
* Train the LBPH Face Recognizer.
* Save the trained model as `Trainner.yml`.

### 3. Recognize Student

* Open the webcam.
* Detect the student's face.
* Recognize the student using the trained model.
* Detect the uniform color.
* Mark attendance automatically.

### 4. View Attendance

Displays all attendance records, including:

* Student ID
* Student Name
* In Date
* In Time
* Out Date
* Out Time
* Uniform Status

---

## 👕 Uniform Detection

The system checks the student's uniform by analyzing the lower portion of the webcam image.

**Required Uniform:**

* Blue Shirt
* Red Tie

If both colors are detected:

```
Uniform : YES
```

Otherwise:

```
Uniform : NO
```

---

## 📊 Attendance Format

| ID  | Name | In Date    | In Time  | Out Date   | Out Time | Uniform |
| --- | ---- | ---------- | -------- | ---------- | -------- | ------- |
| 101 | John | 2026-06-19 | 09:15:02 | 2026-06-19 | 04:30:11 | YES     |

---

## ▶️ How to Run

### Install Required Libraries

```bash
pip install opencv-contrib-python
pip install numpy
pip install pandas
pip install pillow
```

### Run the Project

```bash
python main.py
```

---

## 💻 GUI Buttons

### ADD

Captures student face images.

### TRAIN

Trains the face recognition model.

### RECOGNISE

Recognizes students and marks attendance.

### CLEAR

Clears the input fields.

### VIEW

Displays attendance records.

---

## 📷 Face Recognition

The project uses the **Local Binary Pattern Histogram (LBPH)** algorithm available in OpenCV.

Advantages:

* Fast recognition
* Good accuracy
* Works well with grayscale images
* Suitable for real-time attendance systems

---

## 📈 Future Enhancements

* Database integration (MySQL)
* Email attendance reports
* QR Code support
* Face mask detection
* Multiple camera support
* Deep learning-based face recognition
* Mobile application integration
* Cloud storage for attendance records

---

## 👨‍💻 Author

**Md Hassan Raza**

BCA Graduate

Python Developer | Face Recognition | Computer Vision

---

## 📄 License

This project is developed for educational and learning purposes.
