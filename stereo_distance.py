# stereo_distance.py
# 双目测距模块

import cv2
import numpy as np
import pandas as pd
import math

def load_camera_params(file_path=None):
    """加载相机参数，可以从文件读取或使用硬编码值
    
    Args:
        file_path: 相机参数文件路径（Excel文件）
        
    Returns:
        tuple: (左相机内参，左相机畸变系数，右相机内参，右相机畸变系数，旋转矩阵，平移向量)
    """
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

def init_stereo_rectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, R, T, size=(640, 480)):
    """初始化立体校正
    
    Args:
        left_camera_matrix: 左相机内参
        left_distortion: 左相机畸变系数
        right_camera_matrix: 右相机内参
        right_distortion: 右相机畸变系数
        R: 旋转矩阵
        T: 平移向量
        size: 图像尺寸
        
    Returns:
        tuple: 包含校正需要的所有参数
    """
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        left_camera_matrix, left_distortion,
        right_camera_matrix, right_distortion,
        size, R, T)
    
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
    
    return {
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'validPixROI1': validPixROI1, 'validPixROI2': validPixROI2,
        'left_map1': left_map1, 'left_map2': left_map2,
        'right_map1': right_map1, 'right_map2': right_map2
    }

def create_stereo_matcher(num_disparities=160, block_size=15):
    """创建立体匹配器
    
    Args:
        num_disparities: 视差范围
        block_size: 匹配块大小
        
    Returns:
        stereo: 配置好的StereoBM匹配器
    """
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
    stereo.setPreFilterCap(31)
    stereo.setBlockSize(block_size)
    stereo.setMinDisparity(4)
    stereo.setNumDisparities(num_disparities)
    stereo.setTextureThreshold(50)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)
    return stereo

def compute_disparity(img_left, img_right, stereo_matcher, rectify_params):
    """计算视差图
    
    Args:
        img_left: 左图像
        img_right: 右图像
        stereo_matcher: 立体匹配器
        rectify_params: 校正参数
        
    Returns:
        tuple: (视差图，归一化用于显示的视差图)
    """
    # 将图像转换为灰度
    if len(img_left.shape) == 3:
        img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    else:
        img_left_gray = img_left
        
    if len(img_right.shape) == 3:
        img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        img_right_gray = img_right
    
    # 校正图像
    img_left_rectified = cv2.remap(
        img_left_gray, 
        rectify_params['left_map1'], 
        rectify_params['left_map2'], 
        cv2.INTER_LINEAR
    )
    img_right_rectified = cv2.remap(
        img_right_gray, 
        rectify_params['right_map1'], 
        rectify_params['right_map2'], 
        cv2.INTER_LINEAR
    )
    
    # 设置ROI
    stereo_matcher.setROI1(rectify_params['validPixROI1'])
    stereo_matcher.setROI2(rectify_params['validPixROI2'])
    
    # 计算视差
    disparity = stereo_matcher.compute(img_left_rectified, img_right_rectified)
    
    # 归一化视差图以便显示
    disp_normalized = cv2.normalize(
        disparity, disparity, 
        alpha=100, beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )
    
    return disparity, disp_normalized, img_left_rectified, img_right_rectified

def compute_3d_points(disparity, Q, scale=16):
    """计算三维点云
    
    Args:
        disparity: 视差图
        Q: 视差到深度变换矩阵
        scale: 比例因子
        
    Returns:
        threeD: 三维点云
    """
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * scale
    return threeD

def calculate_distance(point_3d):
    """计算3D点到相机的欧氏距离
    
    Args:
        point_3d: 3D点坐标 [x, y, z]
        
    Returns:
        distance: 距离（米）
    """
    x, y, z = point_3d
    return math.sqrt(x**2 + y**2 + z**2) / 1000.0  # 毫米转米

def get_distance_at_point(threeD, x, y):
    """获取指定点的距离
    
    Args:
        threeD: 三维点云
        x, y: 图像上的点坐标
        
    Returns:
        distance: 距离（米）
    """
    try:
        point_3d = threeD[y][x]
        distance = calculate_distance(point_3d)
        return distance
    except Exception as e:
        print(f"无法计算点 ({x}, {y}) 的距离: {e}")
        return float('nan')