import cv2, time, winsound, os
from datetime import datetime
from PIL import Image

while True:
    faceCascadeDIR = input("Enter the directory for haarcascade_frontalface_default.xml: ")

    if os.path.isdir(faceCascadeDIR):
        break
    elif faceCascadeDIR == '':
        faceCascadeDIR = os.getcwd()
        break
    else:
        print("***Invalid directory***")

while True:
    imageSaveDIR = input("Enter the directory in which to save the detected face: ")

    if os.path.isdir(imageSaveDIR):
        break
    elif imageSaveDIR == '':
        imageSaveDIR = os.getcwd()
        break
    else:
        print("***Invalid directory***")

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(os.path.join(faceCascadeDIR, "haarcascade_frontalface_default" + "." + "xml"))

faceNotFound = True

while faceNotFound:
    
    check, frame = video.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        grayImg,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
    )

    cv2.imshow('Capturing', frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        now = datetime.now()
        winsound.Beep(2000, 400)
        img = Image.fromarray(frame)
        img.save(os.path.join(imageSaveDIR, f"{now.strftime('%d-%b-%y %I%M %p')}" + "." + "jpg"), "JPEG", optimize=True, progressive=True)
        faceNotFound = False

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows
