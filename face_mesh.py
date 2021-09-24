import cv2
import time
import numpy as np
import mediapipe as mp

blank_image = np.zeros((500,500,3), np.uint8)
for i in range(500):
    for j in range(500):
        blank_image[i][j] = 255

f = open("Text/area.txt", "+w")
m = open("Text/mode.txt", "+w")
wav_path = '/home/pi/Documents/Drowsy_detect/alarm.wav'
dem = 0
gat_num = 0
ty_le_tb_mat = 0
trang_thai_trc = 0
dem_gat = 0
tt_mat = ''
tt_mat_trc = ''
tt_dau = ''
Drowsy_mode = ''
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
cap = cv2.VideoCapture(0)
cap.set(3, 1080)   # float `width`
cap.set(4, 720)
canh_bao = False

while True:
    ret, img = cap.read()
    ih, iw = img.shape[0], img.shape[1]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results:
        face = []
        Left_eye = []
        Right_eye = []
        try:
            for face_lms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, face_lms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

        except:
            print(results)
    cv2.imshow("Face Mesh", img)
    key = cv2.waitKey(50)
    if key == ord('q'):
        break


f.close()
cap.release()
cv2.destroyAllWindows()

