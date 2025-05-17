# bottle_detector.py
# 瓶子检测模块

import cv2
import numpy as np
from rknn.api import RKNN

# 初始化参数
MODEL_SIZE = (640, 640)  # 模型输入尺寸

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# 后处理参数
OBJ_THRESH = 0.2
NMS_THRESH = 0.5
color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def init_model(model_path):
    """初始化RKNN模型
    
    Args:
        model_path: RKNN模型路径
        
    Returns:
        初始化好的RKNN模型实例
    """
    rknn = RKNN()
    print('--> 加载RKNN模型')
    if rknn.load_rknn(model_path) != 0:
        print('加载RKNN模型失败')
        return None
    
    print('--> 初始化运行时环境')
    if rknn.init_runtime(target='rk3588', device_id=0) != 0:
        print('初始化运行时环境失败!')
        return None
    
    return rknn

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def letter_box(im, new_shape, pad_color=(255, 255, 255), info_need=False):
    """调整图像尺寸并填充"""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    if info_need is True:
        return im, ratio, (dw, dh)
    else:
        return im

def filter_boxes(boxes, box_confidences, box_class_probs):
    """筛选符合阈值的边界框"""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """非极大值抑制"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def softmax(x, axis=None):
    """Softmax函数"""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    """Distribution Focal Loss (DFL)"""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    """处理边界框坐标"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([MODEL_SIZE[1] // grid_h, MODEL_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def post_process(input_data):
    """YOLO检测结果后处理"""
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    
    if not nclasses and not nscores:
        return None, None, None
    
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    
    return boxes, classes, scores

def detect_bottles(image, rknn_model):
    """检测图像中的瓶子
    
    Args:
        image: 输入图像
        rknn_model: 初始化好的RKNN模型
        
    Returns:
        bottle_detections: 列表，包含检测到的瓶子信息 (left, top, right, bottom, score, center_x, center_y)
    """
    img = letter_box(image, MODEL_SIZE)
    input_tensor = np.expand_dims(img, axis=0)
    outputs = rknn_model.inference([input_tensor])
    boxes, classes, scores = post_process(outputs)
    
    bottle_detections = []
    if boxes is not None:
        img_h, img_w = image.shape[:2]
        x_factor = img_w / MODEL_SIZE[0]
        y_factor = img_h / MODEL_SIZE[1]
        
        for box, score, cl in zip(boxes, scores, classes):
            if CLASSES[cl] == 'bottle':  # 只保留瓶子类别
                x1, y1, x2, y2 = [int(_b) for _b in box]
                
                left = int(x1 * x_factor)
                top = int(y1 * y_factor)
                right = int(x2 * x_factor)
                bottom = int(y2 * y_factor)
                
                # 计算瓶子中心点
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                bottle_detections.append((left, top, right, bottom, score, center_x, center_y))
    
    return bottle_detections

def draw_bottle(image, left, top, right, bottom, score):
    """在图像上绘制瓶子的边界框"""
    bottle_class_id = CLASSES.index('bottle')
    color = color_palette[bottle_class_id]
    
    # 绘制边界框
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
    
    # 添加标签
    label = f"bottle: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10
    cv2.rectangle(image, (int(label_x), int(label_y - label_height)), 
                 (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
    cv2.putText(image, label, (int(label_x), int(label_y)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_bottle_with_distance(image, left, top, right, bottom, score, distance):
    """在图像上绘制瓶子的边界框和距离信息"""
    bottle_class_id = CLASSES.index('bottle')
    color = color_palette[bottle_class_id]
    
    # 绘制边界框
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
    
    # 添加标签和距离信息
    if np.isnan(distance):
        label = f"bottle: {score:.2f}"
    else:
        label = f"bottle: {score:.2f}, 距离: {distance:.2f}m"
    
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10
    cv2.rectangle(image, (int(label_x), int(label_y - label_height)), 
                 (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
    cv2.putText(image, label, (int(label_x), int(label_y)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_fps(image, fps):
    """在图像上绘制帧率信息"""
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)