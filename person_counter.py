import cv2
from ultralytics import YOLO

# -------------------------------
# SETTINGS
# -------------------------------
CAMERA_INDEX = 1  # USB camera; change to 0 if needed
MODEL_NAME = "yolo26n.pt"
CONFIDENCE = 0.35

WINDOW_NAME = "AI Public Safety - Person Counter"

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

# -------------------------------
# OPEN USB CAMERA
# -------------------------------
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: USB camera could not be opened.")
    print("Try CAMERA_INDEX = 0 or 2.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------------------
# FULL-SCREEN WINDOW
# -------------------------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("Camera started.")
print("Press Q or ESC to exit.")

# -------------------------------
# MAIN LOOP
# -------------------------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Detect people (class 0) and track them
        results = model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        person_count = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        person_count += 1

                # Draw YOLO boxes and labels
                frame = result.plot()

        # Display person count
        cv2.rectangle(
            frame,
            (10, 10),
            (360, 65),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Person Count: {person_count}",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Q or ESC closes the window
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed.")