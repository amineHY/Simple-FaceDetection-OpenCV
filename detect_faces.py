import numpy as np
import argparse
import cv2


param_confidence = 0.5  # minimum probability to filter weak detections

# Caffe pre-trained model and Caffe 'deploy' prototxt file
model_caffemodel = r"model/res10_300x300_ssd_iter_140000.caffemodel"
model_prototxt = r"model/deploy.prototxt.txt"

print("[INFO] loading ...{0:s} and {1:s}".format(
    model_prototxt, model_caffemodel))
net = cv2.dnn.readNetFromCaffe(
    prototxt=model_prototxt, caffeModel=model_caffemodel)


print("[INFO] load the input image and construct an input blob for the image")
# by resizing to a fixed 300x300 pixels and then normalizing it
image_file = 'images/image1.jpg'
image = cv2.imread(filename=image_file)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0,
                             size=(300, 300), mean=(104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence associated with the prediction
    confidence = detections[0, 0, i, 2]

    if confidence > param_confidence:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box 
        text = "face {:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(img=image, text=text, org=(startX, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 0), thickness=2)
        
# Display results
cv2.imshow("Output", image)
cv2.waitKey(0)
print("[INFO] Done !\n")