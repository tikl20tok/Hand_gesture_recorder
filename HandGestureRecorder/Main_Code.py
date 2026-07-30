import cv2
import time
import math
import threading
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# SIẾT DEBOUNCE: Phải giữ đúng cử chỉ 15 frames liên tiếp (~0.5s)
STABLE_FRAME_THRESHOLD = 15 

# THƯ MỤC & CẤU HÌNH DỌN DẸP
RECORDS_DIR = "foldervideo"
DELETE_OLDER_THAN_DAYS = 16 # Số ngày muốn giữ lại video, cũ hơn sẽ bị xóa

def cleanup_old_records(folder_path, days_limit):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[*] Đã tạo thư mục lưu trữ: {folder_path}/")
        return

    now_time = time.time()
    cutoff_seconds = days_limit * 86400  # 1 ngày = 86400 giây
    deleted_count = 0

    print(f"[*] Đang kiểm tra và dọn dẹp video cũ trong '{folder_path}/' (Hạn: > {days_limit} ngày)...")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".mp4"):
            file_age = now_time - os.path.getmtime(file_path)
            if file_age > cutoff_seconds:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"  [x] Đã xóa video hết hạn: {file_name}")
                except Exception as e:
                    print(f"  [!] Lỗi khi xóa {file_name}: {e}")

    if deleted_count == 0:
        print("[✓] Không có video nào hết hạn cần xóa.")
    else:
        print(f"[✓] Đã dọn dẹp xong {deleted_count} video cũ!")

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 5000000:
        print("[*] Đang tải lại mô hình MediaPipe Tasks chuẩn (~9.3MB)...")
        req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("[✓] Tải thành công hand_landmarker.task!")

# ==========================================
# KHỐI 1: LUỒNG ĐỌC CAMERA (720p)
# ==========================================
class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# KHỐI 2: TOÁN CỬ CHỈ SIÊU NGHIÊM NGẠC
# ==========================================
def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_gesture_hardcore(landmarks):
    wrist = landmarks[0]
    thumb_tip, thumb_ip, thumb_mcp = landmarks[4], landmarks[3], landmarks[2]
    index_tip, index_pip, index_mcp = landmarks[8], landmarks[6], landmarks[5]
    middle_tip, middle_pip, middle_mcp = landmarks[12], landmarks[10], landmarks[9]
    ring_tip, ring_pip, ring_mcp = landmarks[16], landmarks[14], landmarks[13]
    pinky_tip, pinky_pip, pinky_mcp = landmarks[20], landmarks[18], landmarks[17]

    middle_open = (middle_tip.y < middle_pip.y < middle_mcp.y) and (dist_2d(middle_tip, wrist) > dist_2d(middle_mcp, wrist))
    ring_open   = (ring_tip.y < ring_pip.y < ring_mcp.y) and (dist_2d(ring_tip, wrist) > dist_2d(ring_mcp, wrist))
    pinky_open  = (pinky_tip.y < pinky_pip.y < pinky_mcp.y) and (dist_2d(pinky_tip, wrist) > dist_2d(pinky_mcp, wrist))

    index_closed  = index_tip.y > index_pip.y
    middle_closed = middle_tip.y > middle_pip.y
    ring_closed   = ring_tip.y > ring_pip.y
    pinky_closed  = pinky_tip.y > pinky_pip.y

    thumb_index_dist = dist_2d(thumb_tip, index_tip)

    if thumb_index_dist < 0.028 and middle_open and ring_open and pinky_open:
        return "OK"

    thumb_vertical_straight = (thumb_mcp.y - thumb_tip.y) > 0.12
    if thumb_vertical_straight and index_closed and middle_closed and ring_closed and pinky_closed:
        if dist_2d(thumb_tip, index_mcp) > 0.15:
            return "THUMB_UP"

    return "NONE"

# ==========================================
# KHỐI 3: MAIN ENGINE (DIRECT TO DISK)
# ==========================================
def main():
    ensure_model_exists()

    # Dọn dẹp video cũ trong foldervideo ngay khi bật app
    cleanup_old_records(RECORDS_DIR, DELETE_OLDER_THAN_DAYS)

    print("[*] Đang bật Cam 720p...")
    cam = CameraStream(src=0, width=1280, height=720).start()
    time.sleep(1.0)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.9,
        min_hand_presence_confidence=0.9
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # DIRECT-TO-DISK CONTROL
    is_recording = False
    video_writer = None
    recorded_frames_count = 0
    
    last_detected_gesture = "NONE"
    gesture_hold_counter = 0

    print("\n=== ENGINE DIRECT-TO-DISK (KHÔNG TỐN RAM) ===")
    print(f" Yêu cầu: Giữ cử chỉ CHUẨN TÂM liên tục {STABLE_FRAME_THRESHOLD} frames (~0.5s).")
    print(" 👌 (OK)       : Tạo file & Ghi TRỰC TIẾP xuống SSD")
    print(" 👍 (THUMB_UP) : Đóng file lập tức")
    print(" Nút 'q'       : Thoát chương trình\n")

    window_name = "Hand Gesture Recorder UI (720p)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540) 

    print("\n=== ENGINE SẴN SÀNG ===")

    while True:
        t_start = time.perf_counter()

        ret, frame_720p = cam.read()
        if not ret or frame_720p is None:
            continue

        frame_small = cv2.resize(frame_720p, (640, 360))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)
        raw_gesture = "NONE"

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            raw_gesture = detect_gesture_hardcore(landmarks)

        # --- DEBOUNCE SIẾT 15 FRAMES ---
        if raw_gesture != "NONE" and raw_gesture == last_detected_gesture:
            gesture_hold_counter += 1
        else:
            gesture_hold_counter = 1 if raw_gesture != "NONE" else 0
            last_detected_gesture = raw_gesture

        # KÍCH HOẠT CỬ CHỈ
        if gesture_hold_counter == STABLE_FRAME_THRESHOLD:
            # 1. BẮT ĐẦU QUAY -> MỞ FILE TRỰC TIẾP
            if raw_gesture == "OK" and not is_recording:
                is_recording = True
                recorded_frames_count = 0
                filename = f"rec_{int(time.time())}.mp4"
                file_path = os.path.join(RECORDS_DIR, filename)
                
                h, w = frame_720p.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(file_path, fourcc, TARGET_FPS, (w, h))
                print(f"\n>>> [REC] 👌 Mở file ghi trực tiếp: {file_path}")

            # 2. DỪNG QUAY -> ĐÓNG FILE NGAY TẠI CHỖ
            elif raw_gesture == "THUMB_UP" and is_recording:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print(f"\n>>> [STOP] 👍 Đã chốt và hoàn tất file video!")

        # GHI TRỰC TIẾP KHUNG HÌNH VÀO Ổ CỨNG (KHÔNG LƯU RAM)
        if is_recording and video_writer is not None:
            video_writer.write(frame_720p)
            recorded_frames_count += 1

        # Render UI
        proc_time_ms = (time.perf_counter() - t_start) * 1000
        status_txt = f"REC DIRECT ({recorded_frames_count}f)" if is_recording else "STANDBY"
        color = (0, 0, 255) if is_recording else (0, 255, 0)
        
        if raw_gesture != "NONE":
            bar_width = int((gesture_hold_counter / STABLE_FRAME_THRESHOLD) * 200)
            cv2.rectangle(frame_small, (10, 110), (10 + bar_width, 125), (0, 255, 255), -1)
            cv2.rectangle(frame_small, (10, 110), (210, 125), (255, 255, 255), 1)

        cv2.putText(frame_small, f"Status: {status_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame_small, f"Gesture: {raw_gesture} ({gesture_hold_counter}/{STABLE_FRAME_THRESHOLD}f)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_small, f"Latency: {proc_time_ms:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Cảnh báo trực tiếp khi đang ghi file
        if is_recording:
            cv2.putText(frame_small, "[DANG GHI TRUC TIEP XUONG DISK]", (10, 165), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recorder UI (720p)", frame_small)

        elapsed = time.perf_counter() - t_start
        sleep_ms = max(1, int((FRAME_TIME - elapsed) * 1000))

        key = cv2.waitKey(sleep_ms) & 0xFF
        is_window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        if key == ord('q') or key == ord('Q') or is_window_closed:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print("\n[*] Đã bấm 'q' -> Đã giải phóng file và thoát an toàn!")
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()