# Home-Security-using-Object-Detection
import cv2
import numpy as np
import pyttsx3
import pyaudio
import struct
from twilio.rest import Client
import winsound

count = 0

# Path to the YOLOv3 weights and configuration files
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"

# Check if the files exist
try:
    with open(weights_path) as f:
        pass
    with open(config_path) as f:
        pass
except FileNotFoundError:
    print("One or both of the files does not exist.")
    # Exit the script or handle the error as needed
    exit()

# Load YOLO
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize Twilio client
account_sid = 'AC2c7270071e1b411884f754117de7411c'
auth_token = '593c3ddd0f1f86e9fb50332368ef8f8b'
client = Client(account_sid, auth_token)
from_number = '+16076008309'
to_number = '+917439839069'

# Initialize microphone
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 44100

p = pyaudio.PyAudio()
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def send_sms(message):
    client.messages.create(
        body=message,
        from_=from_number,
        to=to_number
    )

# Initialize background frame for motion detection
global cap
cap = cv2.VideoCapture(0)
_, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (21, 21), 0)

# Initialize sound detection parameters
sound_threshold = 500  # Adjust as needed
previous_sound_level = 0

# Flag to keep track of alarm status
alarm_active = False

# Initialize video recording
recording = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = None  # Initialize VideoWriter object
videowriter = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

def trigger_alarm_sound():
    # Beep sound with frequency 1000Hz for 1 second
    winsound.Beep(1000, 1000)

while True:
    # Read audio data from the microphone
    data = stream.read(chunk)
    data_int = struct.unpack(str(2 * chunk) + 'B', data)
    sound_level = np.mean(data_int)

    if sound_level > 1000:  # Adjust threshold as needed
        print("Suspicious sound detected!")
        speak("Suspicious sound detected!")
        send_sms("Suspicious sound detected!")

    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error reading frame from the camera.")
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            # Speak the detected object label
            speak("Detected " + label)

            # If person detected, trigger alarm and start recording
            if label == "person":
                print("Person detected! Triggering alert and starting recording.")
                speak("Person detected!")
                trigger_alarm_sound()
                send_sms("Person detected: " + label)
                recording = True
                if out is None:
                    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Write frame to video file if recording
    if recording and out is not None:
        # videowriter.write(frame)
        cv2.imwrite(f"{count}_person.jpg", frame)
        count +=1

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
if out is not None:
    out.release()
cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()
