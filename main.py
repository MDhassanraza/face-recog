import tkinter as tk
import cv2, os, csv, time
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from tkinter import ttk


# ===================== COLOR THEME =====================
BG_COLOR = "#0F172A"
CARD_COLOR = "#1E293B"
ACCENT = "#38BDF8"
BTN_ADD = "#22C55E"
BTN_TRAIN = "#6366F1"
BTN_RECOG = "#F59E0B"
BTN_CLEAR = "#64748B"
TEXT_COLOR = "#E5E7EB"

# ===================== PATHS =====================
USER_CSV = "UserDetails/UserDetails.csv"
ATTENDANCE_PATH = "Attendance/attendance.csv"

os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("UserDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)

# ===================== WINDOW =====================
window = tk.Tk()
window.title("Smart Attendance System")
window.geometry("550x360")
window.configure(bg=BG_COLOR)
window.resizable(False, False)

# ===================== HEADER =====================
tk.Label(window, text="SMART ATTENDANCE SYSTEM",
         font=("Segoe UI", 20, "bold"),
         bg=BG_COLOR, fg=ACCENT).pack(pady=8)

tk.Label(window, text="Face Recognition • Uniform Detection",
         font=("Segoe UI", 10),
         bg=BG_COLOR, fg=TEXT_COLOR).pack()

# ===================== CARD =====================
card = tk.Frame(window, bg=CARD_COLOR)
card.place(x=40, y=90, width=440, height=180)

# ===================== INPUTS =====================
tk.Label(card, text="Student ID",
         font=("Segoe UI", 11),
         bg=CARD_COLOR, fg=TEXT_COLOR).place(x=30, y=30)

identry = tk.Entry(card, font=("Segoe UI", 12), width=22)
identry.place(x=150, y=30, height=30)

tk.Label(card, text="Student Name",
         font=("Segoe UI", 11),
         bg=CARD_COLOR, fg=TEXT_COLOR).place(x=30, y=80)

nameentry = tk.Entry(card, font=("Segoe UI", 12), width=22)
nameentry.place(x=150, y=80, height=30)

resultlabel = tk.Label(card, text="",
                       font=("Segoe UI", 10, "bold"),
                       bg=CARD_COLOR, fg="#22C55E")
resultlabel.place(x=150, y=130)

# ===================== UNIFORM DETECTION =====================
def detect_uniform_color(frame):
    h, w, _ = frame.shape
    roi = frame[int(h*0.5):h, int(w*0.25):int(w*0.75)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # BLUE (includes dark blue)
    blue = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))

    # RED (two HSV ranges)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    total_pixels = roi.shape[0] * roi.shape[1]

    blue_ratio = cv2.countNonZero(blue) / total_pixels
    red_ratio = cv2.countNonZero(red) / total_pixels

    if blue_ratio > 0.12 and red_ratio > 0.08:
        return "YES"
    return "NO"

# ===================== ATTENDANCE =====================
def mark_attendance(Id, Name, uniform):
    
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    cols = ["Id", "Name", "inDate", "inTime", "outDate", "outTime", "Uniform"]

    if not os.path.exists(ATTENDANCE_PATH):
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.read_csv(ATTENDANCE_PATH)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

    mask = (df["Id"] == Id) & (df["inDate"] == today)

    if not mask.any():
        df.loc[len(df)] = [Id, Name, today, now, "", "", uniform]
    else:
        idx = df[mask].index[0]
        if df.at[idx, "outTime"] == "":
            df.at[idx, "outDate"] = today
            df.at[idx, "outTime"] = now
            df.at[idx, "Uniform"] = uniform

    df.to_csv(ATTENDANCE_PATH, index=False)

# ===================== ADD FACE =====================
def add():
    Id = identry.get()
    Name = nameentry.get()

    if not (Id.isdigit() and Name.isalpha()):
        resultlabel.config(text="Invalid ID or Name", fg="red")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 7)

        for (x,y,w,h) in faces:
            sample += 1
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            cv2.imwrite(f"TrainingImage/{Name}.{Id}.{sample}.jpg", face)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Capture Faces", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or sample >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(USER_CSV, "a+", newline="") as f:
        csv.writer(f).writerow([Id, Name, time.ctime()])

    resultlabel.config(text="Images Captured Successfully", fg="#22C55E")

# ===================== TRAIN =====================
def get_images(path):
    faces, ids = [], []
    for img in os.listdir(path):
        image = Image.open(os.path.join(path, img)).convert("L")
        faces.append(np.array(image))
        ids.append(int(img.split(".")[1]))
    return faces, ids

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images("TrainingImage")
    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    resultlabel.config(text="Training Completed", fg="#22C55E")

# ===================== RECOGNISE =====================
def recognise():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv(USER_CSV)

    cam = cv2.VideoCapture(0)
    marked = set()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 7)

        uniform = detect_uniform_color(img)
        cv2.putText(img, f"Uniform: {uniform}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        for (x,y,w,h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                name = df.loc[df["Id"] == Id]["Name"].values[0]
                cv2.putText(img, f"{Id}-{name}",
                            (x,y+h), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255,255,255), 2)

                if Id not in marked:
                    mark_attendance(Id, name, uniform)
                    marked.add(Id)
            else:
                cv2.putText(img, "Unknown",
                            (x,y+h), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,0,255), 2)

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# ===================== CLEAR =====================
def clear():
    identry.delete(0, "end")
    nameentry.delete(0, "end")

def view_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        resultlabel.config(text="No attendance data found", fg="red")
        return

    view_win = tk.Toplevel(window)
    view_win.title("Attendance Records")
    view_win.geometry("900x400")
    view_win.configure(bg=BG_COLOR)

    tk.Label(
        view_win,
        text="ATTENDANCE DETAILS",
        font=("Segoe UI", 16, "bold"),
        bg=BG_COLOR,
        fg=ACCENT
    ).pack(pady=10)

    frame = tk.Frame(view_win, bg=CARD_COLOR)
    frame.pack(fill="both", expand=True, padx=20, pady=10)

    tree = ttk.Treeview(frame, show="headings")
    tree.pack(fill="both", expand=True)

    df = pd.read_csv(ATTENDANCE_PATH)

    tree["columns"] = list(df.columns)

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

def hover(btn, enter_color, leave_color):
    btn.bind("<Enter>", lambda e: btn.config(bg=enter_color))
    btn.bind("<Leave>", lambda e: btn.config(bg=leave_color))


# ===================== BUTTONS =====================
def btn(text, color, cmd, x):
    b = tk.Button(
        window,
        text=text,
        command=cmd,
        bg=color,
        fg="white",
        font=("Segoe UI", 11, "bold"),
        bd=0,
        width=10,
        cursor="hand2",
        activeforeground="white"
    )
    b.place(x=x, y=290, height=38)

    # Colorful hover mapping
    hover_map = {
        "ADD": "#16A34A",        # emerald
        "TRAIN": "#4F46E5",      # deep indigo
        "RECOGNISE": "#EA580C",  # vivid orange
        "CLEAR": "#475569",      # slate
        "VIEW": "#0284C7"        # sky blue
    }

    hover(b, hover_map.get(text, color), color)



btn("ADD", BTN_ADD, add, 40)
btn("TRAIN", BTN_TRAIN, train, 140)
btn("RECOGNISE", BTN_RECOG, recognise, 240)
btn("CLEAR", BTN_CLEAR, clear, 340)
btn("VIEW", "#0EA5E9", view_attendance, 440)


window.mainloop()   
