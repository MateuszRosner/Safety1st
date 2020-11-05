import numpy as np
import argparse
import imutils
import time
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

class Detector():
    def __init__(self):

        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-f", "--face", type=str,
            default="face_detector",
            help="path to face detector model directory")
        self.ap.add_argument("-m", "--model", type=str,
            default="mask_detector.model",
            help="path to trained face mask detector model")
        self.ap.add_argument("-c", "--confidence", type=float, default=0.7,
            help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())

        print("[INFO] loading face detector model...")
        self.prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
        self.weightsPath = os.path.sep.join([self.args["face"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        self.maskNet = load_model(self.args["model"])

    def __del__(self):
        cv2.destroyAllWindows()
        self.vs.stop()

    def start_video_stream(self, source):
        self.vs = VideoStream(src=source).start()
        time.sleep(2.0)
        print("Video streaming established")

    def detect_and_predict_mask(self, frame):
        # grab the dimensions of the frame and then construct a blob from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            for f in faces:
                p = (self.maskNet.predict(f))
                preds.append(tuple(p[0]))

        return (locs, preds)

    def process_video(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=800)

        masks = []
        withoutMasks = []

        (locs, preds) = self.detect_and_predict_mask(frame)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask OK" if mask > withoutMask else "No Mask!"
            color = (0, 255, 0) if label == "Mask OK" else (0, 0, 255)

            # include the probability in the label
            label = "{}  {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX - 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            masks.append(mask)
            withoutMasks.append(withoutMask)

        cv2.putText(frame, f"Detected faces: {len(locs)}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Frame", frame)
        return (masks, withoutMasks, frame)