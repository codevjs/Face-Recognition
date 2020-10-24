import cv2
import numpy
import pickle
import requests
from threading import Timer

face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer    = cv2.face.LBPHFaceRecognizer_create()
cap           = cv2.VideoCapture(0)
person        = ""

recognizer.read("trainner.yml")

labels = { "person_name" : 1 }

with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels    = {v:k for k,v in og_labels.items()}

while(True) :

    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casecade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:

        roi_grey  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

       # recognize ? deep learned
        id_, conf = recognizer.predict(roi_grey)

        print(conf)

        if conf >= 45:

            font   = cv2.FONT_HERSHEY_SIMPLEX
            name   = labels[id_]
            color  = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 0.5, color, stroke, cv2.LINE_AA)

            print(name)

            # broadcast google nest
            url   = 'http://192.168.1.100:3000/assistant'
            req   = {
                        "user"      : "Assistant Relay",
                        "command"   : name  + " at the front door",
                        "broadcast" : True
                    }

            if name != person:
                person = name
                requests.post(url, data = req)
                # time = Timer(20, person = "")
                # time.start()
            else:
                print("Orang yang sama")

        img_item = labels[id_] + ".png"
        cv2.imwrite(img_item, roi_color)

        color      = (255, 0, 0) #BGR 0-255
        stroke     = 2
        end_core_x = x + w
        end_core_y = y + h
        cv2.rectangle(frame, (x, y), (end_core_x, end_core_y) ,color, stroke)

    # Display the resulting frame
    imS = cv2.resize(frame, (960, 540))
    cv2.imshow("CCTV", imS)
    cv2.resizeWindow('CCTV', 600,600)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()