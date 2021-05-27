# -*- coding: utf-8 -*-


import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/ObstacleTracking/Model")

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def non_max_suppression_fast(boxes, nmsThreshold):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]+boxes[:,0]
    y2 = boxes[:,3]+boxes[:,1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > nmsThreshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data
    return boxes[pick].astype("int").tolist(), np.array(pick)
    
class YOLO():
    def __init__(self):

        self.confThreshold = 0.5
        self.nmsThreshold = 0.55
        self.inpWidth = 512
        self.inpHeight = 512
        classesFile = "coco.names"
        self.classes = None
        with open(classesFile,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        modelConfiguration = "YoloV3.cfg";
        modelWeights = "yolov3.weights";
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]




    def postprocess(self,frame, outs):

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.    
        #indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold) # Previous Indices
        filtered_boxes = []
        if len(boxes)>0:
            filtered_boxes, indices = non_max_suppression_fast(boxes, self.nmsThreshold) # CALLING THE OTHER NMS FORMULA
            for idx, box in enumerate(filtered_boxes):
                i = indices[idx]
                left, top, width, height = box
                #output_image = self.drawPred(frame,classIds[i], confidences[i], left, top, left + width, top + height)
        
        return frame, filtered_boxes
        
    def inference(self,image):

        #image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        # Remove the bounding boxes with low confidence
        final_frame, boxes = self.postprocess(image, outs)
        return final_frame, boxes