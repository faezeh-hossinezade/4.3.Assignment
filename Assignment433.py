import cv2
import numpy as np



cap=cv2.VideoCapture(0)

def transparent_sticker(sticker):
    rows, cols, _ = sticker.shape
    sticker_ghost = np.zeros(sticker.shape)
    sticker_transparent = np.zeros(sticker.shape)
    sticker_ghost = np.array(sticker_ghost, dtype=np.uint8)
    sticker_transparent = np.array(sticker_transparent, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if sticker[row,col,0] == 255 and sticker[row,col,1] == 255 and sticker[row,col,2] == 255:
                sticker_ghost[row,col] = [0, 0, 0]
            else:
                sticker_ghost[row,col] = sticker[row,col]

            if sticker_ghost[row,col,0] <10 and sticker_ghost[row,col,1] <10 and sticker_ghost[row,col,2] <10:
                sticker_ghost[row,col] = [1, 1, 1]
            else:
                sticker_ghost[row,col] = [0, 0, 0]
            
            if sticker_ghost[row,col,0] == 0 and sticker_ghost[row,col,1] == 0 and sticker_ghost[row,col,2] == 0:
                sticker_transparent[row,col] = sticker[row,col]

        if sticker.shape == (220, 400, 3):
            sticker_ghost[50:150,150:200] = [0, 0, 0]
            sticker_transparent[50:150,150:200] = sticker[50:150,150:200]

    return sticker_ghost, sticker_transparent

img_sticker=cv2.imread("Input/emoji.jpg")
img_stickerrr=cv2.imread("Input/glass.png")
img_stickerr=cv2.imread("Input/smile.jpg")
image_glass_ghost, image_glass_transparent = transparent_sticker(img_stickerrr)
image_smile_ghost, image_smile_transparent = transparent_sticker(img_stickerr)

def sticker_face(image, sticker_ghost, sticker_transparent, face_datector):
    image_gary = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_datector.detectMultiScale(image_gary,1.3)
    for face in faces:
        x, y, w, h = face
        sticker_ghost = cv2.resize(sticker_ghost, [w, h])
        sticker_transparent = cv2.resize(sticker_transparent, [w, h])
        image[y:y+h, x:x+w] = sticker_ghost*image[y:y+h, x:x+w] + sticker_transparent

    return image


def sticker_faces(faces,frame):
    for face in faces:
        x,y,w,h = face
        sticker = cv2.resize(img_sticker,[w,h])
        for i in range(h):
            for j in range(w):
                if sticker[i][j][0] == 201 and sticker[i][j][1] == 174 and sticker[i][j][2] == 255:
                    sticker[i][j] = frame[y+i,x+j]
        frame[y:y+h,x:x+w] = sticker

    return frame

# def sticker_lip_eye(frame,frame_gray):
#     smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#     smiles = smile_detector.detectMultiScale(frame_gray,0,minSize=(70,75),maxSize=(130, 135))

#     for smile in smiles:
#             x,y,w,h = smile
#             sticker_smile = cv2.resize(img_stickerr,[w,h])
#             for i in range(h):
#                 for j in range(w):
#                     if sticker_smile[i][j][0] == 255 and sticker_smile[i][j][1] == 255 and sticker_smile[i][j][2] == 255:
#                         sticker_smile[i][j] = frame[y+i,x+j]
#             frame[y:y+h,x:x+w] = sticker_smile
    
#     eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     eyes = eye_detector.detectMultiScale(frame_gray,1.3,minSize=(50, 50),maxSize=(150, 150)) 
#     for eye in eyes:
#         x,y,w,h = eye
#         p1=x+w//2
#         p2=y+h//2
#         cv2.circle(frame,[p1,p2],h//2,0,5)

#     return frame

def eye_smile(image, sticker_ghost, sticker_transparent, smile_ghost, smile_transparent, eye_datector, smaile_datector):
    image_gary = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    eyes = eye_datector.detectMultiScale(image_gary,1.1)
    smiles = smaile_datector.detectMultiScale(image_gary,1.3)
    
    if len(eyes) == 2:
        eye_x_min = np.minimum(eyes[0,0], eyes[1,0])
        eye_x_max = np.maximum(eyes[0,0], eyes[1,0]) + eyes[1,2]
        eye_y_min = np.minimum(eyes[0,1], eyes[1,1])
        eye_y_max = np.maximum(eyes[0,1], eyes[1,1]) + eyes[1,3]
        sticker_ghost = cv2.resize(sticker_ghost, [eye_x_max - eye_x_min, eye_y_max - eye_y_min])
        sticker_transparent = cv2.resize(sticker_transparent, [eye_x_max - eye_x_min, eye_y_max - eye_y_min])
        image[eye_y_min:eye_y_max, eye_x_min:eye_x_max] = sticker_ghost*image[eye_y_min:eye_y_max, eye_x_min:eye_x_max]//2 + sticker_transparent//2 + image[eye_y_min:eye_y_max, eye_x_min:eye_x_max]//2
    
        for smile in smiles:
            x, y, w, h = smile
            if y > eye_y_max and eye_x_min < x < eye_x_max and (x+w) < eye_x_max and y < (eye_y_max + (eye_x_max - eye_x_min)*.5):
                smile_ghost = cv2.resize(smile_ghost, [w, h])
                smile_transparent = cv2.resize(smile_transparent, [w, h])
                image[y:y+h, x:x+w] = smile_ghost*image[y:y+h, x:x+w] + smile_transparent

    return image

def chess_face(faces, frame):

    for face in faces:
        x,y,w,h=face
        face_image=frame[y:y+h,x:x+w]
        face_image_small=cv2.resize(face_image,[20,20])
        face_image_big=cv2.resize(face_image_small,(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
        frame[y:y+h,x:x+w]=face_image_big
        
    return frame
        



while True:
    _,frame=cap.read()
    face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")
    eye_datector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    smaile_datector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(frame_gray,1.5)
    row,col,band = frame.shape

    cv2.imshow('Default',frame)
    
    if cv2.waitKey(30) & 0xFF == ord ('1'):
       sticker_img=sticker_faces(faces,frame)
       cv2.imshow("chess face",sticker_img)
       cv2.imwrite('Output/Sticker_face.jpg',sticker_img)
        
    if cv2.waitKey(30) & 0xFF == ord('2'):
        # sticker_frame = sticker_lip_eye(faces,frame)
        # cv2.imshow('Face Detection',sticker_frame)
        # cv2.imwrite('Output/Sticker_face.jpg',sticker_frame)
        frame = eye_smile(frame, image_glass_ghost, image_glass_transparent,
        image_smile_ghost, image_smile_transparent, eye_datector, smaile_datector)
        cv2.imshow("eye smile face",frame)
        cv2.imwrite('Output/eye_smile.jpg',frame)
        
        
    if cv2.waitKey(30) & 0xFF == ord ('3'):
       chess=chess_face(faces,frame)
       cv2.imshow("chess face",chess)
       cv2.imwrite('Output/Chess_face.jpg',chess)
       
    if cv2.waitKey(25) & 0xFF == ord('4'):
        
        framelr = np.fliplr(frame)
        frame[:,0:col//2] = frame[:,0:col//2]
        frame[:,col//2:] = framelr[:,col//2:]
        cv2.imshow('Face Detection',frame)
        cv2.imwrite('Output/Mirror_face.jpg',frame)
    
    
