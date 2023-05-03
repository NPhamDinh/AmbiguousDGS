# import libraries
import cv2
from cv2 import VideoWriter
import os
import dlib
from imutils import face_utils
import numpy as np
import face_recognition

dlib.DLIB_USE_CUDA = True

#detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat') #dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

lel = []

b = 0

def crop(path, dst):
    video_capture = cv2.VideoCapture(path)

    height, width = False, False
    while True:

        ret, frame = video_capture.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = frame[:, :, ::-1]
        except:
            break
            
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        
        if len(face_locations) != 1:
            try:
                os.remove(dst)
            except:
                pass
            
            return 0
        
        top, right, bottom, left = face_locations[0]
        face_loc = dlib.rectangle(left, top, right, bottom)

        shape = predictor(gray, face_loc)#rects[0].rect)
        shape = face_utils.shape_to_np(shape)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "mouth":
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = frame[y:y + h, x:x + w]
                roi = cv2.resize(roi, (150, 100), interpolation=cv2.INTER_LINEAR)


                if not height:
                    width, height = 150, 100
                    video = VideoWriter(dst,
                                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                        int(video_capture.get(cv2.CAP_PROP_FPS)), (width, height))
                break
        video.write(roi)
    video_capture.release()

    try:
        video.release()
    except UnboundLocalError:
        pass

for label in os.listdir("./ambiguous"):
    for vid in os.listdir("./ambiguous/" + label):
        target_dir = "./ambiguous_mouth/" + label + "/"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        crop("./ambiguous/" + label + "/" + vid, target_dir + vid[:-3]+"avi")
#crop("input_video.mp4", "output.avi")
