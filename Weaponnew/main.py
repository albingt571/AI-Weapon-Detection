import torch
import yaml
import cv2
import pyttsx3
import threading
import os
from datetime import datetime
import ultralytics

def text_to_speech(frame):
    engine = pyttsx3.init()
    engine.say('Weapon Detected')
    engine.runAndWait()

    # Save the frame as an image with a timestamp
    save_path = 'WeaponDetection Pictures'  # Replace with the desired folder path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f'weapon_detection_{timestamp}.jpg'
    image_path = os.path.join(save_path, image_name)
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")


model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='AIYolov5/content/yolov5/runs/train/yolov5s_results/weights/best.pt',
                       force_reload=True)

with open('AIYolov5/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

classes = data['names']

cap = cv2.VideoCapture(0)
cv2.namedWindow('AI Weapon Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AI Weapon Detection', 800, 600)

while True:
    ret, frame = cap.read()

    results = model(frame)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_idx = result.tolist()

        if confidence > 0.67:
            class_name = classes[int(class_idx)]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

            # Start a new thread for text-to-speech and image capture
            threading.Thread(target=text_to_speech, args=(frame.copy(),)).start()

    cv2.imshow('AI Weapon Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:  # Press Q, q or Esc to quit
        break
    elif key == ord('c') or key == ord('C'):  # Press C or c to close the window
        cv2.destroyAllWindows()
        break
    elif key == ord('m') or key == ord('M'):  # Press M or m to maximize the window
        cv2.setWindowProperty('AI Weapon Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('n') or key == ord('N'):  # Press N or n to minimize the window
        cv2.setWindowProperty('AI Weapon Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
