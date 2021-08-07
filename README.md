# Real-Time-Object-Tracking
**Video Analysis** , which is the primary focus behind this project. The aim is to track the obstacles in each frame.

*The pipeline is as follows Object Detection->Object Tracking(Association)-> Output*

# Object Detection
The tracking will lie heavily on the detection algorithm, so i have used yolov3 as the backbone with OpenCV engine(faster than darknet).



# Obstacle Tracking
Now comes the main part of this project, the idea behind this is to associate bounding boxes from frame t-1 to t. This task has been achieved by the **Hungarian Algorithm**, which is used for association(through a metric) and ID distribution.
<img width="600" alt="Association" src="https://user-images.githubusercontent.com/65212523/119887121-5be62e00-bf51-11eb-8957-61cad0ea7e12.png">

# Metrics
To calculate similarity between two boxes i have considered 3 metrics based off this [paper](https://arxiv.org/pdf/1709.03572.pdf)-:
- IOU Score
- Sanchez Mattila Score
- Exponential Score


# Results


https://user-images.githubusercontent.com/65212523/119888237-bdf36300-bf52-11eb-80a3-32263ff1e1fe.mp4

The id's dont change with the frames and the obstacles have been tracked.

# Problems
- Inference on YOLO is slow, if you have cuda use the following functions
```
self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_GPU)
```
- If the IOU matches and the classes are diffreent, the algorithm wont be able to distingiush between the objects

# Future Works
- Look into Compression techniques such as Pruning and Quantization. Sparse Quantized Yolo models have a very low inference time
- Implement [DeepSort](https://arxiv.org/pdf/1703.07402.pdf) a, CNN based metric which uses as a *Siamese Network* to account for the spatial features as well.
- For more robustness code a **Kalman Filter** to predict the next position of an obstacle along with association.









