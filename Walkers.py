import cv2


# Create our body classifier
bodyClassifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    bodies = bodyClassifier.detectMultiScale(grey, 1.2, 3)

    for x,y,w,h in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
