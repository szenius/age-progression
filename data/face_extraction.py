import cv2
import sys
import glob 

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

files=glob.glob("*.png")  # TODO: change extension accordingly
for file in files:

    # Read the image
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Crop Padding
    left = 5
    right = 5
    top = 5
    bottom = 5

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print(x, y, w, h)

        # Debugging boxes
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        crop  = image[y-top:y+h+bottom, x-left:x+w+right]

        print("cropped_{1}{0}".format(str(file),str(x)))
        cv2.imwrite("cropped_{1}_{0}".format(str(file),str(x)), crop)