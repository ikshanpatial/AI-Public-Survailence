import cv2
import time
import serial
from ultralytics import YOLO

# ==================================
# SETTINGS
# ==================================
CAMERA_INDEX = 1
POSE_MODEL = "yolo11n-pose.pt"
PERSON_MODEL = "yolo26n.pt"

CONFIDENCE = 0.35
ESP32_PORT = "COM5"
BAUD_RATE = 115200

MOVEMENT_THRESHOLD = 35
REQUIRED_FRAMES = 3
ALERT_COOLDOWN = 3
BUZZER_DURATION = 2

WINDOW_NAME = "AI Public Surveillance Camera"

# ==================================
# ESP32 CONNECTION
# ==================================
esp32 = None

def buzzer_on():
    if esp32 and esp32.is_open:
        esp32.write(b"1")

def buzzer_off():
    if esp32 and esp32.is_open:
        esp32.write(b"0")

try:
    esp32 = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    buzzer_off()
    print("ESP32 connected.")
except serial.SerialException as error:
    print("ESP32 connection failed:", error)
    print("Camera will run without the buzzer.")

# ==================================
# LOAD MODELS
# ==================================
print("Loading person detection model...")
person_model = YOLO(PERSON_MODEL)

print("Loading pose detection model...")
pose_model = YOLO(POSE_MODEL)

# ==================================
# OPEN CAMERA
# ==================================
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Camera not opening. Try CAMERA_INDEX = 0.")
    buzzer_off()
    if esp32 and esp32.is_open:
        esp32.close()
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ==================================
# MOVEMENT VARIABLES
# ==================================
previous_points = {}
movement_frames = 0
last_alert_time = 0
buzzer_off_time = 0
alert_active = False

print("Camera started. Press Q or ESC to exit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # ==================================
        # PERSON DETECTION AND COUNTING
        # ==================================
        person_results = person_model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        person_count = 0

        for result in person_results:
            if result.boxes is not None:
                person_count = sum(
                    1 for box in result.boxes
                    if int(box.cls[0]) == 0
                )
                frame = result.plot()

        # ==================================
        # BODY POSE / SKELETON DETECTION
        # ==================================
        pose_results = pose_model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )

        movement_detected = False
        current_points = {}

        for result in pose_results:
            if result.keypoints is None:
                continue

            # Compatible with your installed Ultralytics version
            frame = result.plot()

            points_array = result.keypoints.xy.cpu().numpy()

            for person_index, points in enumerate(points_array):
                # COCO keypoints: 9 = left wrist, 10 = right wrist
                for wrist_id in (9, 10):
                    if wrist_id >= len(points):
                        continue

                    x, y = points[wrist_id]

                    if x <= 0 or y <= 0:
                        continue

                    key = (person_index, wrist_id)
                    current_points[key] = (float(x), float(y))

                    if key in previous_points:
                        old_x, old_y = previous_points[key]

                        distance = (
                            (x - old_x) ** 2 +
                            (y - old_y) ** 2
                        ) ** 0.5

                        if distance > MOVEMENT_THRESHOLD:
                            movement_detected = True

        previous_points = current_points

        # ==================================
        # MOVEMENT ALERT
        # ==================================
        if movement_detected:
            movement_frames += 1
        else:
            movement_frames = 0

        now = time.time()

        if (
            movement_frames >= REQUIRED_FRAMES
            and now - last_alert_time >= ALERT_COOLDOWN
        ):
            print("Movement alert! Please review the video.")

            buzzer_on()
            buzzer_off_time = now + BUZZER_DURATION
            alert_active = True
            last_alert_time = now
            movement_frames = 0

        if alert_active and now >= buzzer_off_time:
            buzzer_off()
            alert_active = False

        # ==================================
        # DISPLAY PERSON COUNT AND STATUS
        # ==================================
        cv2.rectangle(
            frame, (10, 10), (460, 115), (0, 0, 0), -1
        )

        cv2.putText(
            frame,
            f"Person Count: {person_count}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        status = (
            "MOVEMENT ALERT"
            if movement_detected
            else "Monitoring"
        )

        status_color = (
            (0, 0, 255)
            if movement_detected
            else (0, 255, 0)
        )

        cv2.putText(
            frame,
            status,
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    buzzer_off()
    camera.release()
    cv2.destroyAllWindows()

    if esp32 and esp32.is_open:
        esp32.close()

    print("Camera stopped. Buzzer OFF.")