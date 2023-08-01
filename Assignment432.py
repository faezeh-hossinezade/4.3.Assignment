import cv2


img=cv2.imread("Input/cats.jpeg")
cats_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cats_detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalcatface.xml")
faces=face_cats_detector.detectMultiScale(cats_gray)
print("output:",len(faces))

# cv2.imshow("Output/catsnumber",face_cats_detector)
# cv2.waitKey()
