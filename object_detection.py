import cv2
import mediapipe as mp
import time
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.3,
                            min_tracking_confidence=0.3,
                            model_name='Cup') as objectron:
    
    while True:
        ret, img = cap.read()
        start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        result = objectron.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.detected_objects:
            try:
                for detected_objects in result.detected_objects:
                    mp_drawing.draw_landmarks(img, detected_objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(img, detected_objects.rotation, detected_objects.translation)
            except:
                print("fail")
        end = time.time()
        fps = int(1/(end-start))
        cv2.putText(img, str(fps), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('MediaPipe Pose', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
        
