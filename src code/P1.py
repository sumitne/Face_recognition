# creating face data set form web cam

import cv2
import numpy as np

# cascade classifier
# address to classifier file , change accordingly
face_classifier = cv2.CascadeClassifier('C:/Users/SUMITN~1/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_extractor(img): # will extract face from feed and return cropeed face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return None

    for(x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:   # will provide web cam feed to the face_extractor function
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200), )   # will lower the resolution of image
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # path to folder where all images will be stored , change accordingly
        file_name_path = 'C:/Users/Sumit Negi/Documents/Python Scripts/facial recognition/facedata/face'+str(count)+'.jpg'  # location for face data
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print('face not found')
        pass
    if cv2.waitKey(1) == 13 or count == 200:
        break

cap.release()
cv2.destroyAllWindows()
print('Samples Collection Complete')








