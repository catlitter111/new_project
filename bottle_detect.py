import cv2
import numpy as np
from rknn.api import RKNN
import time

# 初始化参数
CAMERA_ID = 21  # 摄像头设备号
RKNN_MODEL = "yolo11n.rknn"
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letter_box(im, new_shape, pad_color=(255, 255, 255), info_need=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border

    if info_need is True:
        return im, ratio, (dw, dh)
    else:
        return im

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
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
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
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
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    # Distribution Focal Loss (DFL)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
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

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
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

def draw_fps(image, fps):
    """
    在图像上绘制帧率信息。
    Args:
        image: 输入图像。
        fps: 当前帧率。
    """
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_bottle(image, left, top, right, bottom, score):
    """在图像上绘制瓶子的边界框。
    Args:
        image: 输入图像。
        left, top, right, bottom: 边界框坐标。
        score: 置信度分数。
    """
    # 获取bottle类别的索引
    bottle_class_id = CLASSES.index('bottle')
    
    # 获取bottle类别的颜色
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

def detect_bottles(image, rknn_model):
    """
    检测图像中的瓶子。
    
    Args:
        image: 输入图像 (numpy数组，BGR格式)
        rknn_model: 初始化好的RKNN模型
        
    Returns:
        列表，每个元素是一个元组 (left, top, right, bottom, score)，表示检测到的瓶子坐标和置信度
    """
    # 预处理图像
    img = letter_box(image, MODEL_SIZE)
    input_tensor = np.expand_dims(img, axis=0)
    
    # 运行推理
    outputs = rknn_model.inference([input_tensor])
    
    # 后处理结果
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
                
                bottle_detections.append((left, top, right, bottom, score))
    
    return bottle_detections

if __name__ == '__main__':
    # 创建并加载RKNN模型
    rknn = RKNN()
    print('--> Load RKNN model')
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print('Load RKNN model failed')
        exit()

    # 初始化运行时环境
    print('--> Init runtime environment')
    if rknn.init_runtime(target='rk3588', device_id=0) != 0:
        print('Init runtime failed!')
        exit()

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        # 初始化帧率相关变量
        start_time = time.time()
        frame_count = 0
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # 检测瓶子
            bottle_detections = detect_bottles(frame, rknn)
            
            # 只绘制瓶子的检测结果
            for left, top, right, bottom, score in bottle_detections:
                draw_bottle(frame, left, top, right, bottom, score)
                print(f'Bottle detected: box coordinates [left, top, right, bottom]: [{left}, {top}, {right}, {bottom}], score: {score:.2f}')
            
            # 计算并显示帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            draw_fps(frame, fps)
            
            # 显示结果
            cv2.imshow('RKNN Bottle Detection', frame)

            # 按Q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()