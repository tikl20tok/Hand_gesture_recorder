import cv2
import time

# Khởi tạo camera với DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Ép Codec MJPEG để đọc 1080p 30fps
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Không mở được Camera!")
    exit()

print("✅ Đã bật Cam 1080p! Nhấn 'q' để thoát test.")

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lỗi đọc frame!")
        break

    # Tính FPS thực tế
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Lấy kích thước thực tế camera đáp ứng
    h, w, _ = frame.shape

    # Bắn thông số lên màn hình
    cv2.putText(frame, f"Res: {w}x{h} | FPS: {fps:.1f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize nhẹ để preview không bị tràn màn hình Desktop
    preview = cv2.resize(frame, (960, 540))
    cv2.imshow("Test Cam 1080p", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()