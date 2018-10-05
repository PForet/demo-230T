import cv2
import sys
import PIL
from models import Model
from utils import draw_expressions, draw_gender, draw_age, draw_smiley

model = Model()
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Extract image and classify
        imgs = frame[y:y+h, x:x+w]
        imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
        imgs = PIL.Image.fromarray(imgs)
        expressions, gender, age = model.classify(imgs)
        # Plot the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        draw_gender(gender, frame, x+w+10, y+20)
        draw_age(age, frame, x+w+10, y+40)
        x,y = draw_expressions(expressions, frame, x+w+10,y+80)
        try:
            frame = draw_smiley(expressions, frame, x+20, y+20)
        except:
            pass
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()