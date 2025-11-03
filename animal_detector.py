import numpy as np
import time
import cv2
import os
import threading
import smtplib
import mimetypes
from email.message import EmailMessage
from playsound import playsound

# === Alert Sound ===
def alert():
    threading.Thread(
        target=playsound,
        args=(r'C:\Users\lavan\OneDrive\Desktop\minor project\alarm.wav',),
        daemon=True
    ).start()

# === Email Sending ===
def send_email(label, filepath):
    Sender_Email = "@gmail.com"
    Receiver_Email = "@gmail.com"
    Password = ''  # Enter app password here

    msg = EmailMessage()
    msg['Subject'] = "Animal Detected"
    msg['From'] = Sender_Email
    msg['To'] = Receiver_Email
    msg.set_content('An animal has been detected.')

    with open(filepath, 'rb') as f:
        image_data = f.read()
        image_type = mimetypes.guess_type(f.name)[0]
        image_name = os.path.basename(f.name)

    msg.add_attachment(image_data, maintype='image', subtype=image_type.split('/')[1], filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(msg)

def async_email(label, filepath):
    threading.Thread(target=send_email, args=(label, filepath), daemon=True).start()

# === YOLO Configuration ===
args = {"confidence": 0.5, "threshold": 0.3}
flag = True

labelsPath = r"C:\Users\lavan\OneDrive\Desktop\minor project\yolo-coco\coco (1).names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'zebra', 'bear', 'giraffe']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = r"C:\Users\lavan\OneDrive\Desktop\minor project\yolo-coco\yolov3-tiny (1).weights"
configPath = r"C:\Users\lavan\OneDrive\Desktop\minor project\yolo-coco\yolov3-tiny (1).cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
layers = net.getUnconnectedOutLayers()
ln = [ln[i - 1] if len(layers.shape) == 1 else ln[i[0] - 1] for i in layers]

# === Input Source ===
input_type = "webcam"  # Change to "webcam", "image", or "video"
input_path = ""  # Only needed for image or video

(W, H) = (None, None)

def process_frame(frame):
    global flag, W, H

    if frame is None:
        return

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label = LABELS[classIDs[i]]

            if label in final_classes:
                if flag:
                    alert()
                    flag = False

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(label, confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    os.makedirs('images', exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f'images/{label}_{timestamp}.png'
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] Saved image with detection: {filename}")
                    async_email(label, filename)

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(label, confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        flag = True

    cv2.imshow("Output", frame)
    if input_type == "image":
        cv2.waitKey(0)

# === Input Handling ===
if input_type == "image":
    frame = cv2.imread(input_path)
    if frame is None:
        print("[ERROR] Could not load image.")
        exit()
    process_frame(frame)

elif input_type in ["webcam", "video"]:
    vs = cv2.VideoCapture(0 if input_type == "webcam" else input_path)
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        process_frame(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    vs.release()

cv2.destroyAllWindows()
