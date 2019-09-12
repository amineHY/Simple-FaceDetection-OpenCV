from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


param_confidence = 0.5  # minimum probability to filter weak detections

# Caffe pre-trained model and Caffe 'deploy' prototxt file
model_caffemodel = r"model/res10_300x300_ssd_iter_140000.caffemodel"
model_prototxt = r"model/deploy.prototxt.txt"

print("[INFO] loading ...{0:s} and {1:s}".format(
    model_prototxt, model_caffemodel))
net = cv2.dnn.readNetFromCaffe(
    prototxt=model_prototxt, caffeModel=model_caffemodel)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
cap = VideoStream(src=0).start()
time.sleep(2.0)

# Save video + detection on the disk
save_output = True
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = None

while True:  # loop over the frames from the video stream
    frame = cap.read()
    frame = imutils.resize(frame, width=400)


    if writer is  None:
        (h, w) = frame.shape[:2]
        writer = cv2.VideoWriter("output.avi", fourcc, 24, (w, h), True)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0,
                                 size=(300, 300), mean=(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    tic = time.time()

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < param_confidence:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "face {:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    toc = time.time()
    print("time %f ms"%((toc-tic)*1000))

    # show the output frame
    cv2.imshow("Frame", frame)

    # write video on disk
    writer.write(frame) if save_output else 0

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.stop()
writer.release() if save_output else 0
cv2.destroyAllWindows()
print("[INFO] Done !\n")
