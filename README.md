# Simple-FaceDetection-OpenCV
This repository contains a simple face detector from a given image or a stream video that can run when an object is moving in the video.


# Demo
![demo.gif](demo.gif)


# Usage

## Run the script
```bash
python face_detection.py
```

## Run the AI-lab and start your development
If you don't have OpenCV installed on your machine, you can use AI-lab, a complete development envirnement to run your computer vision application.

Simply install `docker-ce` and then run in the terminal:

``` bash
xhost +
docker run -it --rm 
--runtime=nvidia 
-v $(pwd):/workspace \
-w /workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-p 8888:8888 -p 6006:6006 aminehy/ai-lab:latest
```
