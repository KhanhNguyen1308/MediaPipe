import cv2
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
resize = False
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    s = time.time()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        print(hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
    if resize:
      image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    fps = 1/(time.time()-s)
    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0), 2)
    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
    if key == ord('r'):
      resize = True
cap.release()