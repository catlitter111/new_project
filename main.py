# main.py
# 整合瓶子检测和双目测距的主程序

import cv2
import numpy as np
import time

# 导入自定义模块
import bottle_detector as bd
import stereo_distance as sd

def main():
    # 相机、模型路径配置
    CAMERA_ID = 1  # 双目相机设备号
    RKNN_MODEL = "./my-main/yolo11n.rknn"  # RKNN模型路径
    CAMERA_PARAMS_FILE = "out.xls"  # 相机参数文件
    
    # 图像尺寸配置
    SIZE = (640, 480)  # 单目图像尺寸
    
    # 加载相机参数
    print("--> 加载相机参数")
    left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, R, T = sd.load_camera_params(CAMERA_PARAMS_FILE)
    
    # 初始化立体校正
    print("--> 初始化立体校正")
    rectify_params = sd.init_stereo_rectify(
        left_camera_matrix, left_distortion,
        right_camera_matrix, right_distortion,
        R, T, SIZE
    )
    
    # 创建立体匹配器
    stereo_matcher = sd.create_stereo_matcher(num_disparities=160, block_size=15)
    
    # 初始化RKNN模型
    print("--> 初始化RKNN模型")
    rknn_model = bd.init_model(RKNN_MODEL)
    if rknn_model is None:
        print("模型初始化失败，退出程序")
        return
    
    # 打开双目相机
    print("--> 打开双目相机")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, 1280)  # 设置宽度为1280（左右相机各640）
    cap.set(4, 480)   # 设置高度为480
    
    if not cap.isOpened():
        print("无法打开相机，退出程序")
        return
    
    try:
        # 初始化帧率相关变量
        start_time = time.time()
        frame_count = 0
        
        print("--> 开始处理视频流")
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法接收帧，退出循环")
                break
            
            # 分割左右相机图像
            frame_left = frame[0:480, 0:640]
            frame_right = frame[0:480, 640:1280]
            
            # 计算视差图和3D点云
            disparity, disp_normalized, img_left_rectified, img_right_rectified = sd.compute_disparity(
                frame_left, frame_right, stereo_matcher, rectify_params
            )
            
            # 将校正后的灰度图转换回彩色图用于检测和显示
            frame_left_rectified = cv2.cvtColor(img_left_rectified, cv2.COLOR_GRAY2BGR)
            
            # 计算三维点云
            threeD = sd.compute_3d_points(disparity, rectify_params['Q'], scale=16)
            
            # 在左图上检测瓶子
            bottle_detections = bd.detect_bottles(frame_left_rectified, rknn_model)
            
            # 显示检测结果并计算距离
            for left, top, right, bottom, score, cx, cy in bottle_detections:
                # 获取瓶子中心点的距离
                distance = sd.get_distance_at_point(threeD, cx, cy)
                print(f'瓶子检测: 坐标 [{left}, {top}, {right}, {bottom}], 分数: {score:.2f}, 距离: {distance:.2f}m')
                
                # 在图像上绘制瓶子和距离信息
                bd.draw_bottle_with_distance(frame_left_rectified, left, top, right, bottom, score, distance)
            
            # 计算并显示帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            bd.draw_fps(frame_left_rectified, fps)
            
            # 显示结果
            cv2.imshow("原始左图", frame_left)
            cv2.imshow("原始右图", frame_right)
            cv2.imshow("瓶子检测结果", frame_left_rectified)
            cv2.imshow("视差图", disp_normalized)
            
            # 按Q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下Q键，退出程序")
                break
    
    finally:
        # 释放资源
        print("--> 释放资源")
        cap.release()
        cv2.destroyAllWindows()
        if rknn_model:
            rknn_model.release()

# 程序入口
if __name__ == "__main__":
    main()
    