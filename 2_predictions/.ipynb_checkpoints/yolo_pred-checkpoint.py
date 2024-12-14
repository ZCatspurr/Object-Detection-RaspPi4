#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os 
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
    # Real-time predictions
        with open(data_yaml, 'r') as f:
            yaml_data = yaml.load(f, Loader = SafeLoader)
        
        self.labels = yaml_data['names']
        self.nc = yaml_data['nc']
        #print(labels)
        
        #Loading YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image, inp_wid=640):
        
        ratio = image.shape[1] / image.shape[0]
        width = 500
        height = int(width / ratio)
        img = cv2.resize(image, (width, height))

        row, col, d = img.shape
        
        # contort image to rectanglular array 
        max_rowcol = max(row,col)
        inp_image = np.zeros((max_rowcol,max_rowcol,3), dtype = np.uint8)
        inp_image[0:row, 0:col] = img
        
        blob = cv2.dnn.blobFromImage(inp_image, (1/255), (inp_wid, inp_wid), swapRB = True, crop = False)
        self.yolo.setInput(blob)
        predict = self.yolo.forward()
        
        
        #print(predict.shape)
        
        detections = predict[0]
        dimensions = []
        confidences = []
        classes = []
        img_w, img_h = inp_image.shape[:2]
        x_factor = img_w/inp_wid
        y_factor = img_h/inp_wid
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confid in array is in index 4
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax() 
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    
                    box = np.array([left, top, width, height])
                    confidences.append(confidence)
                    dimensions.append(box)
                    classes.append(class_id)
                    
        if not dimensions:
            print("No objs detected.")
            return img
        
        dimensions_np = np.array(dimensions).tolist()
        confidences_np = np.array(confidences).tolist()

        #if len(dimensions_np) > 0:
        index_positions = cv2.dnn.NMSBoxes(dimensions_np, confidences_np, 0.25, 0.45).flatten()
            
        for index in index_positions:
            x, y, w, h = dimensions_np[index]
            bbox_conf = int(confidences_np[index] * 100)
            classes_id = classes[index]
            class_name = self.labels[classes_id]
            colors = self.gen_color(class_id)
        
            text = f'{class_name}: {bbox_conf}%'
            cv2.rectangle(img, (x,y), (x+w, y+h), colors, 2)
            cv2.rectangle(img, (x,y-30), (x+w, y), colors, -1)
            
            cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,0,0), 1)

        return img

        # if more than one class for image detection was generated RGB
    def gen_color(self, ID):
        np.random.seed(5)
        color = np.random.randint(100,255,size = (self.nc, 3)).tolist()
        return tuple(color[ID])
            
        #cv2.imshow('original', image)
        #cv2.imshow('yolo_pred', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()