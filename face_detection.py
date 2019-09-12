#####################################################################
# Import Libriries
#####################################################################
import cv2
import numpy as np
import time
import imutils


#####################################################################
print("\n[INFO] Load pre-trained Caffe model\n")
#####################################################################
# Caffe pre-trained model and Caffe 'deploy' prototxt file
model_caffemodel = r"model/res10_300x300_ssd_iter_140000.caffemodel"
model_prototxt = r"model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(
    prototxt=model_prototxt, caffeModel=model_caffemodel)
param_confidence = 0.5  # minimum probability to filter weak detections

print("[INFO] loading ...{0:s} and {1:s}".format(
    model_prototxt, model_caffemodel))


#####################################################################
print("\n[INFO] Read frames from the webcam / file\n")
#####################################################################
input_file = 0  # 'videos/test_fire_2.mp4'

if isinstance(input_file, str):
    video_source = input_file
    input_file_name = input_file[:-4]
    video = cv2.VideoCapture(video_source)
else:
    video_source = 0
    input_file_name = "videos/webcam"
    video = cv2.VideoCapture(video_source)

time.sleep(2)

if video.isOpened() == False:
    print("[INFO] Unable to read the camera feed")

#####################################################################
# Background Extraction
#####################################################################
# Subtractors
knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)

# Motion detection parameters
percentage = 0.1  # percent
thresholdCount = 1500
movementText = "Motion detected"
textColor = (255, 255, 255)
titleTextPosition = (50, 50)
titleTextSize = 1.2
motionTextPosition = (20,20)
frameIdx = 0


#####################################################################
# Write video settings: Save video + detection on the disk
#####################################################################
save_output = True
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None


print("[INFO] Processing...(Press q to stop)")

while(1):
    # Return Value and the current frame
    ret, frame = video.read()
    frameIdx += 1
    print('[INFO] Frame Number: %d' % (frameIdx))

    #  Check if a current frame actually exist
    if not ret:
        break

    frame = imutils.resize(frame, width=400)

    if writer is None:
        (frame_height, frame_width) = frame.shape[:2]
        output_file_name = input_file_name + \
            '_motion_detection_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer = cv2.VideoWriter(
            output_file_name, fourcc, 20.0, (frame_width, frame_height))

        output_motion_file_name = input_file_name + \
            '_motion_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer_motion = cv2.VideoWriter(
            output_motion_file_name, fourcc, 20.0, (frame_width, frame_height), 0)

        # print information
        pixel_total = frame_height * frame_width
        thresholdCount = (percentage * pixel_total) / 100

        print('[INFO] frame_height={}, frame_width={}'.format(
            frame_height, frame_width))
        print('[INFO] Number of pixels of the frame: {}'.format(pixel_total))
        print('[INFO] Number of pixels to trigger Detection ({}%) : {}'.format(percentage,
                                                                               thresholdCount))

    # Perform Movement Detection: KNN
    #####################################################################
    tic = time.time()
    knnMask = knnSubtractor.apply(frame)
    toc = time.time()

    knnPixelCount = np.count_nonzero(knnMask)
    knnPixelPercentage = (knnPixelCount*100.0)/pixel_total
    print('[INFO] Processing time (Movement Detection): {0:2.2f} ms'.format(
        (toc-tic)*1000))
    print('[INFO] Percentage of Moving Pixel: {0:2.4f} % ({1:d})'.format(
        knnPixelPercentage, knnPixelCount))

    if (knnPixelCount > thresholdCount) and (frameIdx > 1):
        cv2.putText(knnMask, movementText+': {} pixels'.format(knnPixelCount), motionTextPosition,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Perform Face Detection
        #####################################################################
        # Pre-processing of the input frame
        image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0,
                                     size=(300, 300), mean=(104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and  predictions
        tic = time.time()
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections and display overlay
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Ignore detection bellow the confidence threshold
            if confidence < param_confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            h = frame_height
            w = frame_width
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "face {:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        toc = time.time()
        print("time %f ms" % ((toc-tic)*1000))

    # Display Results
    #####################################################################
    cv2.imshow('Face Detection ({} x {})'.format(
        frame_height, frame_width), frame)
    cv2.imshow('Movement: KNN', knnMask)

    cv2.moveWindow('Face Detection ({} x {})'.format(
        frame_height, frame_width), 50, 50)
    cv2.moveWindow('Movement: KNN',  frame_width, 50)

    # Record Video
    writer.write(frame) if save_output else 0
    writer_motion.write(knnMask) if save_output else 0

    # if the `q` key was pressed, break from the loop
    #####################################################################
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
