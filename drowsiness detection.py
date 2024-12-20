import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import threading

# เริ่มต้นเสียงเตือน
mixer.init()
sound = mixer.Sound('alarm.wav')

# โหลด Classifier สำหรับตรวจจับหน้าและดวงตา
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# ตั้งชื่อผลลัพธ์
lbl = ['Close', 'Open']

# โหลดโมเดลที่เทรนแล้ว
model = load_model('models/cnncat2.h5')

# ตั้งค่าพื้นที่ทำงาน
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# ตัวแปรเริ่มต้น
score = 0
thicc = 2
popup_shown = False  # ตรวจสอบว่าแสดงป๊อบอัพหรือไม่
sound_playing = False  # ตรวจสอบว่าเสียงกำลังเล่นอยู่หรือไม่
popup_triggered = False  # ตรวจสอบว่าป๊อบอัพถูกแสดงหรือยัง

def show_popup():
    """ฟังก์ชันแสดงป๊อบอัพ"""
    global popup_shown, popup_triggered
    os.system("osascript -e 'tell app \"System Events\" to display dialog \"คะแนนถึง 30 แล้ว! กด OK เพื่อเริ่มใหม่\" buttons {\"OK\"} default button \"OK\"'")
    popup_shown = False  # ตั้งให้พร้อมแสดงป๊อบอัพใหม่
    popup_triggered = False  # รีเซ็ตสถานะป๊อบอัพ
    reset_program()  # รีเซ็ตโปรแกรมหลังจากปิดป๊อบอัพ

def reset_program():
    """ฟังก์ชันรีเซ็ตโปรแกรม"""
    global score, thicc, sound_playing
    score = 0  # รีเซ็ตคะแนน
    thicc = 2  # รีเซ็ตความหนาของกรอบ
    if sound_playing:
        sound.stop()  # หยุดเสียงเตือน
        sound_playing = False  # อัปเดตสถานะเสียง

def run_program():
    """ฟังก์ชันหลักของโปรแกรม"""
    global score, thicc, popup_shown, sound_playing, popup_triggered
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # กำหนดค่าเริ่มต้นของ rpred และ lpred
        rpred = [1]  # ค่าเริ่มต้นคือ "Open"
        lpred = [1]  # ค่าเริ่มต้นคือ "Open"

        # ตรวจจับตาขวา
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)
            break

        # ตรวจจับตาซ้าย
        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict(l_eye)
            break

        # ตรวจสอบสถานะของตา
        if np.argmax(rpred) == 0 and np.argmax(lpred) == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0

        cv2.putText(frame, 'Score: ' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # หากคะแนนถึง 30 และยังไม่ได้แสดงป๊อบอัพ
        if score >= 30 and not popup_shown:
            popup_shown = True  # ป้องกันการแสดงป๊อบอัพซ้ำ
            popup_triggered = True  # บันทึกสถานะว่าป๊อบอัพถูกแสดง
            threading.Thread(target=show_popup).start()  # แสดงป๊อบอัพในเธรดใหม่

        # หากคะแนนเกิน 15 และเสียงยังไม่ได้เล่น หรือหากป๊อบอัพแสดงแล้ว
        if score >= 15 or popup_triggered:
            if not sound_playing:
                sound.play(-1)  # เล่นเสียงเตือนวนลูป
                sound_playing = True  # อัปเดตสถานะเสียง
        elif sound_playing and score < 15 and not popup_triggered:
            sound.stop()  # หยุดเสียงถ้าคะแนนลดลงต่ำกว่า 15
            sound_playing = False  # อัปเดตสถานะเสียง

        # หากคะแนนเกิน 15 ให้เปลี่ยนความหนาของกรอบ
        if score > 15:
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# เรียกใช้ฟังก์ชันการทำงาน
run_program()
