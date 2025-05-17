import cv2
import numpy as np
from rknn.api import RKNN
import time
import math
import pandas as pd

# 初始化参数
RKNN_MODEL = "./my-main/yolo11n.rknn"
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

# 相机参数 - 可以从文件读取或使用硬编码值
def load_camera_params(file_path=None):
    """加载相机参数，可以从文件读取或使用硬编码值"""
    if file_path:
        try:
            df = pd.read_excel(file_path, header=None)
            
            left_camera_matrix = np.array(df.iloc[0:3, 1:4], dtype=np.float64)
            left_distortion = np.array(df.iloc[5, 1:6], dtype=np.float64).reshape(1, 5)
            right_camera_matrix = np.array(df.iloc[6:9, 1:4], dtype=np.float64)
            right_distortion = np.array(df.iloc[11, 1:6], dtype=np.float64).reshape(1, 5)
            T = np.array(df.iloc[12, 1:4], dtype=np.float64)
            R = np.array(df.iloc[13:16, 1:4], dtype=np.float64)
            
            print("已从文件加载相机参数")
        except Exception as e:
            print(f"无法从文件加载相机参数: {e}")
            print("使用硬编码的相机参数")
            # 使用硬编码的相机参数
            left_camera_matrix = np.array([[479.511022870591, -0.276113089875797, 325.165562307888],
                                        [0., 482.402195086215, 267.117105422009],
                                        [0., 0., 1.]])
            left_distortion = np.array([[0.0544639674308284, -0.0266591889115199, 0.00955609439715649, -0.0026033932373644, 0]])
            right_camera_matrix = np.array([[478.352067946262, 0.544542937907123, 314.900427485172],
                                            [0., 481.875120562091, 267.794159848602],
                                            [0., 0., 1.]])
            right_distortion = np.array([[0.069434162778783, -0.115882071309996, 0.00979426351016958, -0.000953149415242267, 0]])
            R = np.array([[0.999896877234412, -0.00220178317092368, -0.0141910904351714],
                        [0.00221406478831849, 0.999997187880575, 0.00084979294881938],
                        [0.0141891794683169, -0.000881125309460678, 0.999898940295571]])
            T = np.array([[-60.8066968317226], [0.142395217396486], [-1.92683450371277]])
    else:
        # 使用硬编码的相机参数
        left_camera_matrix = np.array([[479.511022870591, -0.276113089875797, 325.165562307888],
                                    [0., 482.402195086215, 267.117105422009],
                                    [0., 0., 1.]])
        left_distortion = np.array([[0.0544639674308284, -0.0266591889115199, 0.00955609439715649, -0.0026033932373644, 0]])
        right_camera_matrix = np.array([[478.352067946262, 0.544542937907123, 314.900427485172],
                                        [0., 481.875120562091, 267.794159848602],
                                        [0., 0., 1.]])
        right_distortion = np.array([[0.069434162778783, -0.115882071309996, 0.00979426351016958, -0.000953149415242267, 0]])
        R = np.array([[0.999896877234412, -0.00220178317092368, -0.0141910904351714],
                    [0.00221406478831849, 0.999997187880575, 0.00084979294881938],
                    [0.0141891794683169, -0.000881125309460678, 0.999898940295571]])
        T = np.array([[-60.8066968317226], [0.142395217396486], [-1.92683450371277]])
    
    return left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, R, T

# YOLO检测相关函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letter_box(im, new_shape, pad_color=(255, 255, 255), info_need=False):
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

def draw_fps(image, fps):
    """在图像上绘制帧率信息"""
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_bottle_with_distance(image, left, top, right, bottom, score, distance):
    """在图像上绘制瓶子的边界框和距离信息"""
    bottle_class_id = CLASSES.index('bottle')
    color = color_palette[bottle_class_id]
    
    # 绘制边界框
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
    
    # 添加标签和距离信息
    label = f"bottle: {score:.2f}, 距离: {distance:.2f}m"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10
    cv2.rectangle(image, (int(label_x), int(label_y - label_height)), 
                 (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
    cv2.putText(image, label, (int(label_x), int(label_y)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def detect_bottles(image, rknn_model):
    """检测图像中的瓶子，返回边界框和置信度"""
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

def calculate_distance(point_3d):
    """计算3D点到相机的欧氏距离"""
    x, y, z = point_3d
    return math.sqrt(x**2 + y**2 + z**2) / 1000.0  # 毫米转米

if __name__ == '__main__':
    # 加载相机参数
    left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, R, T = load_camera_params('out.xls')
    
    # 设置双目相机的尺寸
    size = (640, 480)
    
    # 进行立体校正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        left_camera_matrix, left_distortion,
        right_camera_matrix, right_distortion,
        size, R, T)
    
    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
    
    # 创建并加载RKNN模型
    rknn = RKNN()
    print('--> 加载RKNN模型')
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print('加载RKNN模型失败')
        exit()
    
    # 初始化运行时环境
    print('--> 初始化运行时环境')
    if rknn.init_runtime(target='rk3588', device_id=0) != 0:
        print('初始化运行时环境失败!')
        exit()
    
    # 打开双目相机
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)  # 设置宽度为1280
    cap.set(4, 480)   # 设置高度为480
    
    if not cap.isOpened():
        print("无法打开相机")
        exit()
    
    try:
        # 初始化帧率相关变量
        start_time = time.time()
        frame_count = 0
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法接收帧")
                break
            
            # 分割左右相机图像
            frame_left = frame[0:480, 0:640]
            frame_right = frame[0:480, 640:1280]
            
            # 转换为灰度图
            imgL_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            imgR_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            # 校正图像
            img_left_rectified = cv2.remap(imgL_gray, left_map1, left_map2, cv2.INTER_LINEAR)
            img_right_rectified = cv2.remap(imgR_gray, right_map1, right_map2, cv2.INTER_LINEAR)
            
            # 将灰度图转换回彩色图用于检测和显示
            frame_left_rectified = cv2.cvtColor(img_left_rectified, cv2.COLOR_GRAY2BGR)
            
            # 使用BM算法计算视差
            numberOfDisparities = 160
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
            stereo.setROI1(validPixROI1)
            stereo.setROI2(validPixROI2)
            stereo.setPreFilterCap(31)
            stereo.setBlockSize(15)
            stereo.setMinDisparity(4)
            stereo.setNumDisparities(numberOfDisparities)
            stereo.setTextureThreshold(50)
            stereo.setUniquenessRatio(15)
            stereo.setSpeckleWindowSize(100)
            stereo.setSpeckleRange(32)
            stereo.setDisp12MaxDiff(1)
            
            # 计算视差图
            disparity = stereo.compute(img_left_rectified, img_right_rectified)
            
            # 归一化视差图以便显示
            disp_normalized = cv2.normalize(disparity, disparity, alpha=100, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 计算三维坐标
            threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
            threeD = threeD * 16  # 根据实际情况调整
            
            # 在左图上检测瓶子
            bottle_detections = detect_bottles(frame_left_rectified, rknn)
            
            # 显示检测结果并计算距离
            for left, top, right, bottom, score, cx, cy in bottle_detections:
                # 获取瓶子中心点的3D坐标
                try:
                    point_3d = threeD[cy][cx]
                    # 计算距离
                    distance = calculate_distance(point_3d)
                    print(f'瓶子检测: 坐标 [{left}, {top}, {right}, {bottom}], 分数: {score:.2f}, 距离: {distance:.2f}m')
                    # 在图像上绘制瓶子和距离信息
                    draw_bottle_with_distance(frame_left_rectified, left, top, right, bottom, score, distance)
                except Exception as e:
                    print(f"计算距离时出错: {e}")
                    # 如果无法计算距离，仍然绘制瓶子但不显示距离
                    draw_bottle_with_distance(frame_left_rectified, left, top, right, bottom, score, float('nan'))
            
            # 计算并显示帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            draw_fps(frame_left_rectified, fps)
            
            # 显示结果
            cv2.imshow("原始左图", frame_left)
            cv2.imshow("原始右图", frame_right)
            cv2.imshow("瓶子检测结果", frame_left_rectified)
            cv2.imshow("视差图", disp_normalized)
            
            # 按Q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()

        