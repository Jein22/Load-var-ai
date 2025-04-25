import os
import threading
import time
import tkinter.filedialog as filedialog
from tkinter import messagebox
import customtkinter as ctk
import cv2
import numpy as np
import requests
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# إعدادات المظهر
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# تحميل نماذج
model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# إعداد الواجهة
app = ctk.CTk()
app.geometry("414x896")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.title("AI-Driven VAR Integration in Football")

# متغيرات
video_path = ""
cap = None
pause = False
stop_video = False
video_running = False
ball_positions = []
pass_moment_detected = False
offside_count = 0
frame_counter = 0
total_frames = 0
ball_class_id = 32
player_class_id = 0
confidence_threshold = 0.4
playback_speed = 1.0
save_folder = "offside_snapshots"
os.makedirs(save_folder, exist_ok=True)
auto_report = False

# دالة تحليل الصور بـ BLIP
def analyze_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        return description
    except Exception as e:
        messagebox.showerror("خطأ", f"خطأ في تحليل الصورة: {str(e)}")
        return f"خطأ: {str(e)}"

# دالة اكتشاف خطوط الملعب
def detect_field_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
    offside_line_x = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < abs(y1 - y2):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                offside_line_x = (x1 + x2) // 2
                print(f"Offside line detected at x={offside_line_x}")
    else:
        print("No lines detected")
    return frame, offside_line_x

# دالة التحقق من التسلل
def check_offside(player_boxes, offside_line_x, ball_center, frame):
    print(f"Player boxes: {len(player_boxes)}, Offside line: {offside_line_x}, Ball center: {ball_center}")
    if len(player_boxes) < 2 or offside_line_x is None or ball_center is None:
        print("Offside check skipped: insufficient data")
        return False, None
    player_boxes.sort(key=lambda box: box[1] + box[3])
    last_defender_x = player_boxes[-2][0] + player_boxes[-2][2] / 2
    print(f"Last defender x: {last_defender_x}")
    for (x, y, w, h) in player_boxes:
        player_right_edge = x + w / 2
        tolerance = frame.shape[1] * 0.05
        print(f"Checking player at x={player_right_edge}, tolerance={tolerance}")
        if player_right_edge > max(last_defender_x, offside_line_x) + tolerance:
            print("Offside detected!")
            return True, (x, y, w, h)
    return False, None

# دالة حساب سرعة الكرة
def calculate_speed(positions):
    distances = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distance)
    return np.mean(distances) if distances else 0

# دالة إرسال تنبيه التسلل
def send_offside_alert(description):
    try:
        url = 'http://127.0.0.1:5000/offside'  # تغيير العنوان إلى localhost
        data = {'message': f"🚨 حالة تسلل: {description}"}
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            status_label.configure(text="تم إرسال إشعار التسلل!")
        else:
            status_label.configure(text=f"فشل إرسال الإشعار: {response.status_code}")
    except requests.exceptions.RequestException as e:
        status_label.configure(text="فشل إرسال الإشعار: الخادم غير متاح")
# دالة توليد تقرير نصي
def generate_report():
    try:
        report = f"تقرير حالات التسلل ({time.strftime('%Y-%m-%d %H:%M:%S')}):\n"
        report += f"إجمالي حالات التسلل: {offside_count}\n\n"
        for i, file in enumerate(sorted(os.listdir(save_folder))):
            if file.endswith(".jpg"):
                timestamp = file.split("_")[1]
                description = analyze_image(os.path.join(save_folder, file))
                report += f"حالة التسلل {i + 1} (الوقت: {timestamp}):\n"
                report += f"  - الوصف: {description}\n"
                report += "  - الصورة محفوظة في: " + os.path.join(save_folder, file) + "\n\n"
        with open("offside_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        messagebox.showinfo("نجاح", "تم توليد التقرير النصي في offside_report.txt")
    except Exception as e:
        messagebox.showerror("خطأ", f"خطأ في إنشاء التقرير: {str(e)}")

# دالة توليد تقرير PDF
def generate_pdf_report():
    try:
        pdf_file = "offside_report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        pdfmetrics.registerFont(TTFont('Arabic', 'Amiri-Regular.ttf'))
        c.setFont("Arabic", 12)
        y = 750
        c.drawString(50, y, f"تقرير حالات التسلل ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        y -= 20
        c.drawString(50, y, f"إجمالي حالات التسلل: {offside_count}")
        y -= 30
        for i, file in enumerate(sorted(os.listdir(save_folder))):
            if file.endswith(".jpg"):
                timestamp = file.split("_")[1]
                description = analyze_image(os.path.join(save_folder, file))
                c.drawString(50, y, f"حالة التسلل {i + 1} (الوقت: {timestamp})")
                y -= 20
                c.drawString(50, y, f"الوصف: {description}")
                y -= 20
                img_path = os.path.join(save_folder, file)
                try:
                    img = ImageReader(img_path)
                    c.drawImage(img, 50, y - 100, width=200, height=112, preserveAspectRatio=True)
                    y -= 120
                except Exception as e:
                    c.drawString(50, y, f"تعذر تحميل الصورة: {str(e)}")
                    y -= 20
                if y < 50:
                    c.showPage()
                    c.setFont("Arabic", 12)
                    y = 750
        c.save()
        messagebox.showinfo("نجاح", f"تم توليد تقرير PDF في {pdf_file}")
    except Exception as e:
        messagebox.showerror("خطأ", f"خطأ في إنشاء تقرير PDF: {str(e)}")

# دالة تشغيل الفيديو
def play_video():
    global cap, pause, stop_video, ball_positions, pass_moment_detected, frame_counter, offside_count, total_frames, video_running
    if not video_path or cap is not None:
        video_running = False
        return
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("خطأ", "تعذر فتح ملف الفيديو!")
            return
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        progress_slider.set(0)
        status_label.configure(text=f"تم تحميل الفيديو | الإطار: {frame_counter}/{int(total_frames)}")
        while cap.isOpened():
            if stop_video:
                break
            if pause:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            progress_slider.set(frame_counter / total_frames)
            status_label.configure(text=f"الإطار: {frame_counter}/{int(total_frames)} | سرعة: {playback_speed:.1f}x")
            frame, offside_line_x = detect_field_lines(frame)
            results = model.track(frame, persist=True, classes=[player_class_id, ball_class_id], conf=confidence_threshold)
            player_boxes = []
            ball_center = None
            for i, box in enumerate(results[0].boxes.xyxy):
                cls = int(results[0].boxes.cls[i])
                x1, y1, x2, y2 = map(int, box[:4])
                print(f"Detected class: {cls}, Box: ({x1}, {y1}, {x2}, {y2})")
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                if cls == player_class_id:
                    player_boxes.append((x, y, w, h))
                elif cls == ball_class_id:
                    ball_center = (x, y)
                    ball_positions.append(ball_center)
                    if len(ball_positions) > 10:
                        ball_positions.pop(0)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            annotated_frame = results[0].plot()
            if len(ball_positions) >= 5:
                speed = calculate_speed(ball_positions[-5:])
                ball_near_player = False
                if ball_center:
                    for (px, py, pw, ph) in player_boxes:
                        if abs(px - ball_center[0]) < 150 and abs(py - ball_center[1]) < 150:
                            ball_near_player = True
                            break
                print(f"Ball speed: {speed}, Near player: {ball_near_player}")
                if speed > 3 and ball_near_player and not pass_moment_detected:
                    print("Pass moment detected!")
                    pass_moment_detected = True
                    cv2.putText(annotated_frame, "لحظة التمرير", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    is_offside, offside_player = check_offside(player_boxes, offside_line_x, ball_center, frame)
                    if is_offside and offside_player:
                        pause = True
                        offside_count += 1
                        offside_label.configure(text=f"عدد حالات التسلل: {offside_count}")
                        x, y, w, h = offside_player
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (0, 0, 255), 3)
                        cv2.putText(annotated_frame, "Offside!", (int(x), int(y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        snapshot_name = os.path.join(save_folder, f"offside_{timestamp}_{offside_count}.jpg")
                        cv2.imwrite(snapshot_name, frame)
                        description = analyze_image(snapshot_name)
                        offside_desc_label.configure(text=f"الوصف: {description}")
                        send_offside_alert(description)
                        video_frame.configure(border_color="red", border_width=3)
                        pause_button.configure(text="▶️ استئناف")
                        status_label.configure(text="تم اكتشاف تسلل! اضغط 'تخطي' أو 'استئناف'")
                elif speed < 3 or not ball_near_player:
                    pass_moment_detected = False
            else:
                pass_moment_detected = False
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            aspect_ratio = img.width / img.height
            new_width = 900
            new_height = int(new_width / aspect_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.configure(image=imgtk)
            video_label.imgtk = imgtk
            if app.winfo_exists():
                app.update()
            delay = int(15 / playback_speed)
            time.sleep(delay / 1000)
        if auto_report:
            generate_pdf_report()
    except Exception as e:
        messagebox.showerror("خطأ", f"خطأ أثناء تشغيل الفيديو: {str(e)}")
    finally:
        release_cap()
        video_running = False

# دالة تخطي التسلل
def skip_offside():
    global pause
    pause = False
    pause_button.configure(text="⏸️ إيقاف مؤقت")
    video_frame.configure(border_color="gray", border_width=2)
    status_label.configure(text=f"الإطار: {frame_counter}/{int(total_frames)} | سرعة: {playback_speed:.1f}x")

# دالة تحرير الفيديو
def release_cap():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    status_label.configure(text="انتهى الفيديو!")
    progress_slider.set(0)

# دالة اختيار الفيديو
def browse_video():
    global video_path, frame_counter, offside_count, ball_positions, pass_moment_detected, video_running
    if video_running:
        messagebox.showwarning("تحذير", "جارٍ تشغيل فيديو بالفعل!")
        return
    release_cap()
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("خطأ", "تعذر فتح ملف الفيديو!")
            cap.release()
            return
        cap.release()
        video_path = file_path
        frame_counter = 0
        offside_count = 0
        ball_positions = []
        pass_moment_detected = False
        offside_label.configure(text="عدد حالات التسلل: 0")
        offside_desc_label.configure(text="")
        video_frame.configure(border_color="gray", border_width=2)
        video_running = True
        threading.Thread(target=play_video, daemon=True).start()

# دالة إيقاف مؤقت
def toggle_pause():
    global pause
    pause = not pause
    pause_button.configure(text="⏸️ إوقاف مؤقت" if not pause else "▶️ استئناف")
    video_frame.configure(border_color="gray", border_width=2)
    status_label.configure(text=f"الإطار: {frame_counter}/{int(total_frames)} | سرعة: {playback_speed:.1f}x")

# دالة إيقاف الفيديو
def stop_video_func():
    global stop_video
    stop_video = True
    release_cap()

# دالة إعادة ضبط
def reset_counter():
    global offside_count, frame_counter, ball_positions, pass_moment_detected, stop_video
    offside_count = 0
    frame_counter = 0
    ball_positions = []
    pass_moment_detected = False
    stop_video = True
    progress_slider.set(0)
    offside_label.configure(text="عدد حالات التسلل: 0")
    offside_desc_label.configure(text="")
    video_frame.configure(border_color="gray", border_width=2)
    release_cap()
    status_label.configure(text="تم إعادة ضبط العداد")
    try:
        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))
    except Exception as e:
        messagebox.showerror("خطأ", f"خطأ في حذف اللقطات: {str(e)}")

# دالة تغيير سرعة التشغيل
def update_playback_speed(value):
    global playback_speed
    playback_speed = float(value)
    status_label.configure(text=f"الإطار: {frame_counter}/{int(total_frames)} | سرعة: {playback_speed:.1f}x")

# دالة التنقل في الفيديو
def seek_video(value):
    global frame_counter, pause, ball_positions, pass_moment_detected
    if cap is None:
        return
    pause = True
    pause_button.configure(text="▶️ استئناف")
    target_frame = int(float(value) * total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    frame_counter = target_frame
    ball_positions = []
    pass_moment_detected = False
    status_label.configure(text=f"الإطار: {frame_counter}/{int(total_frames)} | سرعة: {playback_speed:.1f}x")

# دالة الإعدادات
def open_settings():
    settings_window = ctk.CTkToplevel(app)
    settings_window.title("⚙️ إعدادات")
    settings_window.geometry("400x600")
    settings_window.resizable(False, False)
    ctk.CTkLabel(settings_window, text="إعدادات المظهر", font=("Arial", 18, "bold")).pack(pady=10)
    color_mode = ctk.StringVar(value="blue")
    def change_theme(choice):
        ctk.set_default_color_theme(choice)
        for btn in [upload_button, pause_button, stop_button, report_button, pdf_button, reset_button, skip_button, settings_button]:
            btn.configure(fg_color=f"#{choice}99", hover_color=f"#{choice}66")
    theme_menu = ctk.CTkOptionMenu(settings_window, values=["blue", "green", "dark-blue", "purple"],
                                   variable=color_mode, command=change_theme)
    theme_menu.pack(pady=5)
    def toggle_appearance_mode():
        current = ctk.get_appearance_mode()
        new_mode = "dark" if current == "light" else "light"
        ctk.set_appearance_mode(new_mode)
    ctk.CTkButton(settings_window, text="تبديل الوضع (فاتح/داكن)", command=toggle_appearance_mode).pack(pady=10)
    ctk.CTkLabel(settings_window, text="إعدادات الأداء", font=("Arial", 18, "bold")).pack(pady=20)
    def update_confidence(value):
        global confidence_threshold
        confidence_threshold = float(value)
    ctk.CTkLabel(settings_window, text="عتبة الثقة للكشف:").pack(pady=5)
    conf_slider = ctk.CTkSlider(settings_window, from_=0.3, to=0.9, command=update_confidence)
    conf_slider.set(confidence_threshold)
    conf_slider.pack(pady=5)
    ctk.CTkLabel(settings_window, text="سرعة التشغيل الافتراضية:").pack(pady=5)
    speed_slider = ctk.CTkSlider(settings_window, from_=0.5, to=2.0, command=update_playback_speed)
    speed_slider.set(playback_speed)
    speed_slider.pack(pady=5)
    def toggle_resolution():
        current_size = app.geometry()
        new_size = "414x896" if current_size == "1280x720" else "1280x720"
        app.geometry(new_size)
    ctk.CTkButton(settings_window, text="تبديل الحجم (جوال/سطح مكتب)", command=toggle_resolution).pack(pady=10)
    ctk.CTkLabel(settings_window, text="إعدادات التقارير", font=("Arial", 18, "bold")).pack(pady=20)
    def toggle_auto_report():
        global auto_report
        auto_report = not auto_report
    ctk.CTkCheckBox(settings_window, text="توليد تقرير PDF تلقائيًا", command=toggle_auto_report).pack(pady=5)

# دالة إغلاق التطبيق
def on_closing():
    global stop_video, video_running
    stop_video = True
    video_running = False
    release_cap()
    try:
        app.after_cancel(app)
        app.destroy()
    except Exception:
        pass

# إعداد واجهة المستخدم
sidebar = ctk.CTkFrame(app, width=200, corner_radius=10)
sidebar.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
video_frame = ctk.CTkFrame(app, corner_radius=10, fg_color="#1e1e3a", border_color="gray", border_width=2)
video_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack(padx=5, pady=5)
button_frame = ctk.CTkFrame(sidebar, corner_radius=8)
button_frame.pack(pady=20)
offside_label = ctk.CTkLabel(sidebar, text="عدد حالات التسلل: 0", font=("Arial", 14))
offside_label.pack(pady=10)
offside_desc_label = ctk.CTkLabel(sidebar, text="", font=("Arial", 12), wraplength=180)
offside_desc_label.pack(pady=5)
upload_button = ctk.CTkButton(button_frame, text="📂 اختيار فيديو", command=browse_video)
upload_button.pack(pady=5)
pause_button = ctk.CTkButton(button_frame, text="⏸️ إيقاف مؤقت", command=toggle_pause)
pause_button.pack(pady=5)
stop_button = ctk.CTkButton(button_frame, text="⏹️ إيقاف", command=stop_video_func)
stop_button.pack(pady=5)
skip_button = ctk.CTkButton(button_frame, text="⏭️ تخطي التسلل", command=skip_offside)
skip_button.pack(pady=5)
report_button = ctk.CTkButton(button_frame, text="📄 تقرير", command=generate_report)
report_button.pack(pady=5)
pdf_button = ctk.CTkButton(button_frame, text="🧾 PDF", command=generate_pdf_report)
pdf_button.pack(pady=5)
reset_button = ctk.CTkButton(button_frame, text="🔄 إعادة ضبط", command=reset_counter)
reset_button.pack(pady=5)
settings_button = ctk.CTkButton(button_frame, text="⚙️ إعدادات", command=open_settings)
settings_button.pack(pady=5)
progress_label = ctk.CTkLabel(sidebar, text="تقدم الفيديو:", font=("Arial", 12))
progress_label.pack(pady=5)
progress_slider = ctk.CTkSlider(sidebar, from_=0, to=1, command=seek_video, width=180)
progress_slider.set(0)
progress_slider.pack(pady=5)
speed_label = ctk.CTkLabel(sidebar, text="سرعة التشغيل:", font=("Arial", 12))
speed_label.pack(pady=5)
speed_slider = ctk.CTkSlider(sidebar, from_=0.5, to=2.0, command=update_playback_speed, width=180)
speed_slider.set(playback_speed)
speed_slider.pack(pady=5)
status_label = ctk.CTkLabel(sidebar, text="لم يتم تحميل فيديو", font=("Arial", 12))
status_label.pack(pady=10)
app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()

from flask import Flask, request

app = Flask(__name__)

@app.route('/offside', methods=['POST'])
def receive_offside():
    data = request.json
    print(f"Received offside status: {data}")
    return {'status': 'received'}

if __name__ == '__main__':
    app.run(port=5000)
# server.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/offside', methods=['GET'])
def check_offside():
    return jsonify({"status": "offside"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
