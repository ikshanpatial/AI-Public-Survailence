import cv2
import time
import serial
import requests
from ultralytics import YOLO

# ========== CONFIGURATION ==========

CAMERA_INDEX = 1
MODEL_NAME = "yolo11n-pose.pt"

# Change COM5 to your ESP32 port
SERIAL_PORT = "COM5"
BAUD_RATE = 115200

# Add your Telegram bot details
BOT_TOKEN = "PASTE_YOUR_BOT_TOKEN"
CHAT_ID = "PASTE_YOUR_CHAT_ID"

MOVEMENT_THRESHOLD = 35
REQUIRED_FRAMES = 3
BUZZER_DURATION = 2
NOTIFICATION_COOLDOWN = 30

WINDOW_NAME = "AI Public Safety"

# ========== LOAD MODEL ==========

print("Loading YOLO Pose model...")
model = YOLO(MODEL_NAME)

# ========== CONNECT ESP32 ==========

esp32 = None

try:
    esp32 = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1
    )
    time.sleep(2)
    esp32.write(b"0\n")
    print("ESP32 connected.")
except serial.SerialException as error:
    print("ESP32 connection failed:", error)
    print("Check SERIAL_PORT and USB connection.")

# ========== PHONE NOTIFICATION ==========

def send_notification(message):
    if (
        not BOT_TOKEN.startswith("PASTE_")
        and not CHAT_ID.startswith("PASTE_")
    ):
        try:
            url = (
                f"https://api.telegram.org/bot"
                f"{BOT_TOKEN}/sendMessage"
            )

            response = requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "text": message
                },
                timeout=5
            )

            if response.ok:
                print("Telegram notification sent.")
            else:
                print("Telegram error:", response.text)

        except requests.RequestException as error:
            print("Notification failed:", error)
    else:
        print("Configure Telegram token and chat ID first.")

# ========== OPEN USB CAMERA ==========

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("USB camera not found. Try CAMERA_INDEX = 0.")
    if esp32 is not None:
        esp32.close()
    raise SystemExit

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ========== MONITORING ==========

previous_wrists = {}
movement_frames = 0
last_notification = 0
buzzer_until = 0
alert_active = False

print("Monitoring started. Press Q to exit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame error.")
            break

        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            verbose=False
        )

        rapid_movement = False

        for result in results:
            if result.keypoints is None:
                continue

            poses = result.keypoints.xy.cpu().numpy()
            confidence = result.keypoints.conf

            if confidence is not None:
                confidence = confidence.cpu().numpy()

            # Use tracked person IDs when available
            track_ids = None
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()

            for person_index, pose in enumerate(poses):
                person_id = (
                    track_ids[person_index]
                    if track_ids is not None
                    else person_index
                )

                for wrist_index in (9, 10):
                    if confidence is not None:
                        if confidence[person_index][wrist_index] < 0.5:
                            continue

                    x, y = pose[wrist_index]
                    current = (float(x), float(y))
                    key = (person_id, wrist_index)

                    if key in previous_wrists:
                        old_x, old_y = previous_wrists[key]

                        distance = (
                            (current[0] - old_x) ** 2
                            + (current[1] - old_y) ** 2
                        ) ** 0.5

                        if distance > MOVEMENT_THRESHOLD:
                            rapid_movement = True

                    previous_wrists[key] = current

            frame = result.plot()

        # Confirm movement across consecutive frames
        if rapid_movement:
            movement_frames += 1
        else:
            movement_frames = 0

        confirmed_alert = movement_frames >= REQUIRED_FRAMES
        now = time.time()

        if confirmed_alert:
            buzzer_until = max(
                buzzer_until,
                now + BUZZER_DURATION
            )

            if not alert_active:
                print("Possible rapid movement detected.")

            alert_active = True

            if now - last_notification >= NOTIFICATION_COOLDOWN:
                send_notification(
                    "AI Public Safety Alert: "
                    "Possible rapid movement detected. "
                    "Please review the camera feed."
                )
                last_notification = now
        else:
            alert_active = False

        # Control buzzer
        if esp32 is not None:
            buzzer_on = now < buzzer_until
            command = b"1\n" if buzzer_on else b"0\n"

            try:
                esp32.write(command)
            except serial.SerialException:
                print("ESP32 disconnected.")
                esp32.close()
                esp32 = None

        # Display status
        if confirmed_alert:
            status = "POSSIBLE RAPID MOVEMENT"
            color = (0, 0, 255)
        else:
            status = "Monitoring"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    camera.release()

    if esp32 is not None:
        try:
            esp32.write(b"0\n")
            esp32.close()
        except serial.SerialException:
            pass

    cv2.destroyAllWindows()
    print("Monitoring stopped.")