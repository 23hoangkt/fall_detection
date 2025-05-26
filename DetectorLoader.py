# import time
# import torch
# import numpy as np
# import torchvision.transforms as transforms

# from queue import Queue
# from threading import Thread

# from Detection.Models import Darknet
# from Detection.Utils import non_max_suppression, ResizePadding


# class TinyYOLOv3_onecls(object):
#     """Load trained Tiny-YOLOv3 one class (person) detection model.
#     Args:
#         input_size: (int) Size of input image must be divisible by 32. Default: 416,
#         config_file: (str) Path to Yolo model structure config file.,
#         weight_file: (str) Path to trained weights file.,
#         nms: (float) Non-Maximum Suppression overlap threshold.,
#         conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
#         device: (str) Device to load the model on 'cpu' or 'cuda'.
#     """
#     def __init__(self,
#                  input_size=416,
#                  config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
#                  weight_file='Models/yolo-tiny-onecls/best-model.pth',
#                  nms=0.2,
#                  conf_thres=0.45,
#                  device='cuda'):
#         self.input_size = input_size
#         self.model = Darknet(config_file).to(device)
#         self.model.load_state_dict(torch.load(weight_file))
#         self.model.eval()
#         self.device = device

#         self.nms = nms
#         self.conf_thres = conf_thres

#         self.resize_fn = ResizePadding(input_size, input_size)
#         self.transf_fn = transforms.ToTensor()

#     def detect(self, image, need_resize=True, expand_bb=5):
#         """Feed forward to the model.
#         Args:
#             image: (numpy array) Single RGB image to detect.,
#             need_resize: (bool) Resize to input_size before feed and will return bboxs
#                 with scale to image original size.,
#             expand_bb: (int) Expand boundary of the boxs.
#         Returns:
#             (torch.float32) Of each detected object contain a
#                 [top, left, bottom, right, bbox_score, class_score, class]
#             return `None` if no detected.
#         """
#         image_size = (self.input_size, self.input_size)
#         if need_resize:
#             image_size = image.shape[:2]
#             image = self.resize_fn(image)

#         image = self.transf_fn(image)[None, ...]
#         scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

#         detected = self.model(image.to(self.device))
#         detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
#         if detected is not None:
#             detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
#             detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
#             detected[:, 0:4] /= scf

#             detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
#             detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

#         return detected


# class ThreadDetection(object):
#     def __init__(self,
#                  dataloader,
#                  model,
#                  queue_size=256):
#         self.model = model

#         self.dataloader = dataloader
#         self.stopped = False
#         self.Q = Queue(maxsize=queue_size)

#     def start(self):
#         t = Thread(target=self.update, args=(), daemon=True).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped:
#                 return

#             images = self.dataloader.getitem()

#             outputs = self.model.detect(images)

#             if self.Q.full():
#                 time.sleep(2)
#             self.Q.put((images, outputs))

#     def getitem(self):
#         return self.Q.get()

#     def stop(self):
#         self.stopped = True

#     def __len__(self):
#         return self.Q.qsize()




import time
import torch
import numpy as np
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from ultralytics import YOLO

class TinyYOLOv8_onecls(object):
    """Load trained YOLOv8 one class (person) detection model.qq
    Args:
        input_size: (int) Size of input image. Default: 640 (YOLOv8 default),
        weight_file: (str) Path to trained weights file.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=896,  # Tăng từ 640 lên 896
                 #input_size=640,
                 weight_file='Models/yolov8m.pt',  # Default YOLOv8n weights, replace with your trained weights
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size
        self.model = YOLO(weight_file)  # Load YOLOv8 model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.conf_thres = conf_thres

    def detect(self, image, need_resize=True, expand_bb=10):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) Resize to input_size before feed and will return bboxs with scale to image original size.,
            expand_bb: (int) Expand boundary of the boxs.
        Returns:
            (list) Of each detected object contain a [x1, y1, x2, y2, confidence, class] or None if no detection.
        """
        # Resize image if needed
        if need_resize:
            image_resized = cv2.resize(image, (self.input_size, self.input_size))
        else:
            image_resized = image.copy()

        # Perform inference
        results = self.model(image_resized)[0]  # Get the first result (single image)
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                if box.conf[0] >= self.conf_thres:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                    conf = box.conf[0].item()
                    # Assuming class 0 is person (adjust if your model has multiple classes)
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else 0
                    
                    # Scale back if resized
                    if need_resize:
                        h, w = image.shape[:2]
                        scale_x = w / self.input_size
                        scale_y = h / self.input_size
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)

                    # Expand bounding box
                    x1 = max(0, x1 - expand_bb)
                    y1 = max(0, y1 - expand_bb)
                    x2 = min(image.shape[1], x2 + expand_bb)
                    y2 = min(image.shape[0], y2 + expand_bb)

                    detections.append([x1, y1, x2, y2, conf, class_id])

        return detections if detections else None

class ThreadDetection(object):
    # (Keep the existing ThreadDetection class unchanged, just update the model usage)
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model
        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            images = self.dataloader.getitem()
            outputs = self.model.detect(images)
            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()


