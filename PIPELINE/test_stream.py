import cv2
import yaml
import os
import time

# Load config
config_path = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

input_source = cfg.get('INPUT_SOURCE', 0)
# input_source = cfg.get('OUTPUT_STREAM_URL_UDP', 0)

cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print(f"Không thể mở stream: {input_source}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Stream FPS (theoretical): {fps}")

frame_count = 0
fps_counter = 0
fps_display = 0
fps_timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame từ stream hoặc stream đã kết thúc.")
        break
    frame_count += 1
    fps_counter += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_timer = now

    # Ghi FPS lên frame
    show_frame = frame.copy()
    cv2.putText(show_frame, f"FPS: {fps_display}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Test Stream Input", show_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Thoát chương trình...")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Hoàn thành! Đã nhận {frame_count} frames từ stream.") 