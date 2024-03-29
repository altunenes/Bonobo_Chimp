
import cv2
import numpy as np
import glob
import random

#These files must be on your current workspace 
net = cv2.dnn.readNet("yolov4-obj.cfg", "bonoboChimp.weights")

classes = ["bonobo","chimp"]


images_path = glob.glob(r"C:\Users\emportent\Desktop\test\*.jpg")  # image dir for for loop. 

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#you can change colors and frontsizes of rectangles...
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#shuffling images randomly
random.shuffle(images_path)

for img_path in images_path:
    # loading images on given path...
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # blob detecting rectangles
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # rectangle info. You can add various information to your rectangle. 
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #you can also change your confidence
            if confidence > 0.3:
                # after detection, we may print into our console:
                print(class_id)
                print(scores)
                print(confidence)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    # you can change drawing functions
    font = cv2.FONT_HERSHEY_PLAIN
    
    #info of rectangles. You can add confidence; however, sometimes imshow does not work properly with this. I will add this also, you can put it into the after "label" line
    #confidence=confidences[i]
    #then change the cv2.putText line as:
    #cv2.putText(img, label + "" + str(round(confidence, 2)),(x, y + 30), font, 3, color, 2)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    #show images
    cv2.imshow("Image", img)
    #waits until your keyboard response for the passing to the next image
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
