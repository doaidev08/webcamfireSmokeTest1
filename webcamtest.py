import cv2
import streamlit as st
import numpy as np
st.title("Hiển thị AI webcam của bạn")
FRAME_WINDOW = st.image([])
st.subheader("Mời nhập địa chỉ IP và port Webcam của bạn")
st.text("Ví dụ: https://192.168.9.102:8080")
ipcam = st.text_input("")
ipcamvideo = ipcam + "/video"
cap = cv2.VideoCapture(ipcamvideo)
if ipcam:
    st.success("Địa chỉ IP webcam của bạn là:" +" "+ipcam)
colors = np.random.uniform(0, 255, size=(100,3))
run = st.button("Hiển thị webcam")
if run:
    if(ipcam):
        while True:
            _, img = cap.read()
            height, width,channel = img.shape
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # FRAME_WINDOW.image(img) 
            configFile = 'yolov3_testing.cfg'
            weightsFile = 'yolov3_training_last.weights'
            objnames = "obj.names"
            net = cv2.dnn.readNet(configFile,weightsFile)
            
            classes = []
            with open(objnames, "r") as f:
                classes = f.read().splitlines()

            font = cv2.FONT_HERSHEY_PLAIN
            # colors = np.random.uniform(0, 255, size=(100,3))

            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            # cv2.imshow('WebCam detect Smoke-Fire Scientific-Research', img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(img)  
    else: 
        st.warning("Mời nhập địa chỉ IP của bạn")      

