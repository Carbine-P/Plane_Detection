#!/home/orin/anaconda3/envs/ganav/bin/python
import rospy
import cv2
import numpy as np
import open3d as o3d
import json
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Point
from plane_detection.msg import PlaneCorners
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO
import os
import time  
from functools import wraps
import threading

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[Time] {func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

class PlaneDetector:
    def __init__(self):
        rospy.init_node('plane_detector', anonymous=True)


        # 加载相机参数
        camera_json_path = rospy.get_param('~camera_json_path', '/home/orin/PHT/PosDet/data/camera_copy.json')
        with open(camera_json_path, "r") as f:
            cam_data = json.load(f)
        self.K = np.array(cam_data["cam_K"]).reshape(3, 3)
        self.depth_scale = cam_data["depth_scale"]

        # 加载分割模型
        output_dir = "/home/orin/PHT/workspace/src/plane_detection/output"
        model_path = rospy.get_param('~model_path', '/home/orin/PHT/yolo/runs/seg/train8/weights/best.pt')
        # model.to('cuda')  # 或 model.cuda()
        self.target_class = rospy.get_param('~target_class', 'edge')
        self.model = YOLO(model_path)
        rospy.loginfo(f"Loaded YOLO model from: {model_path}")

        #加载检测模型
        self.yolo_model = YOLO("/home/orin/PHT/yolo/runs/detect/train5/weights/best.pt")

        # ========== 统一参数配置区域 ==========
        # 在这里修改所有可配置参数的默认值
        
        # 处理控制参数
        self.process_interval = rospy.get_param('~process_interval', 0.2)  # 处理间隔（秒）
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)  # YOLO置信度阈值
        
        # 保存控制参数  
        self.save_images = rospy.get_param('~save_images', True)  # 是否保存图片
        self.save_pointclouds = rospy.get_param('~save_pointclouds', True)  # 是否保存点云
        
        # 可视化控制参数
        self.enable_visualization = rospy.get_param('~enable_visualization', True)  # 是否启用窗口显示
        
        # 坐标系和单位配置
        self.coordinate_system = rospy.get_param('~coordinate_system', 'right_hand')  # 'camera' 或 'right_hand'
        self.output_unit = rospy.get_param('~output_unit', 'millimeter')  # 'meter', 'centimeter' 或 'millimeter'
        # 单位转换因子：从米转换到目标单位
        if self.output_unit == 'millimeter':
            self.unit_scale = 1000.0  # 米转毫米
        elif self.output_unit == 'centimeter':
            self.unit_scale = 100.0   # 米转厘米
        else:  # meter
            self.unit_scale = 1.0     # 保持米
        
        # ========== 参数验证和日志 ==========
        rospy.logwarn("=" * 60)
        rospy.logwarn("平面检测参数配置:")
        rospy.logwarn(f"  处理间隔: {self.process_interval}秒")
        rospy.logwarn(f"  置信度阈值: {self.confidence_threshold}")
        rospy.logwarn(f"  保存图片: {self.save_images}")
        rospy.logwarn(f"  保存点云: {self.save_pointclouds}")
        rospy.logwarn(f"  可视化显示: {self.enable_visualization}")
        rospy.logwarn(f"  目标类别: {self.target_class}")
        rospy.logwarn(f"  坐标系统: {self.coordinate_system}")
        rospy.logwarn(f"  输出单位: {self.output_unit}")
        rospy.logwarn("=" * 60)
        
        # 创建保存目录
        self.base_output_dir = "/home/orin/PHT/workspace/src/plane_detection/output/"
        self.photo_dir = os.path.join(self.base_output_dir, "photo/")
        self.pcd_dir = os.path.join(self.base_output_dir, "PCD/")
        
        if self.save_images:
            os.makedirs(self.photo_dir, exist_ok=True)
            rospy.loginfo(f"图片保存目录: {self.photo_dir}")
            
        if self.save_pointclouds:
            os.makedirs(self.pcd_dir, exist_ok=True)
            rospy.loginfo(f"点云保存目录: {self.pcd_dir}")

        # 初始化CV Bridge
        self.bridge = CvBridge()

        # 创建发布器
        self.result_pub = rospy.Publisher('/result', PlaneCorners, queue_size=10)

        # 存储最新的同步图像
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.last_processed_time = rospy.Time.now()  # 记录上次处理时间

        # 创建同步订阅器
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        rospy.loginfo("Subscribed to /camera/color/image_raw and /camera/aligned_depth_to_color/image_raw")
        self.ts.registerCallback(self.sync_callback)

        # 设置定时器（10Hz，但实际处理频率由抽帧逻辑控制）
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)


    def sync_callback(self, rgb_msg, depth_msg):
        """存储最新的同步图像"""
        self.latest_rgb_msg = rgb_msg
        self.latest_depth_msg = depth_msg

    def timer_callback(self, event):
        """固定频率检查是否满足抽帧条件"""
        current_time = rospy.Time.now()
        time_diff = (current_time - self.last_processed_time).to_sec()

        # 使用可配置的处理间隔（精确控制）
        if time_diff >= self.process_interval and self.latest_rgb_msg is not None and self.latest_depth_msg is not None:
            rospy.loginfo(f"处理最新帧 (实际间隔: {time_diff:.2f}秒, 设定间隔: {self.process_interval}秒)")
            try:
                self.image_callback(self.latest_rgb_msg, self.latest_depth_msg)

                # 精确更新处理时间，避免累计误差
                self.last_processed_time = rospy.Time.now()  # 使用当前时间而不是检查时间

                # 清空缓存，防止旧帧重复处理
                self.latest_rgb_msg = None
                self.latest_depth_msg = None
            except Exception as e:
                rospy.logerr(f"Processing error: {str(e)}")
                # 即使出错也要更新时间，避免高频重试
                self.last_processed_time = rospy.Time.now()
        
    def image_callback(self, rgb_msg, depth_msg):
        rospy.loginfo("进入 image_callback")
        try:
            start_time = time.time()
            # 转换图像消息
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

            # # -------------------------------
            # # YOLOv8 检测部分
            # # -------------------------------
            # rospy.loginfo("开始执行YOLO目标检测...")
            # results = self.yolo_model(rgb)[0]  # Ultralytics 默认处理单张图像
            # yolo_vis = rgb.copy()

            # rospy.loginfo(f"检测到 {len(results.boxes)} 个目标")
            # if len(results.boxes) >= 1:
            #     # 存储每个检测框中心点及其三维坐标
            #     det_centers_3d = []
            #     label_count = {}
            #     for i, box in enumerate(results.boxes):
            #         x1, y1, x2, y2 = map(int, box.xyxy[0])  # 坐标
            #         conf = float(box.conf[0])
            #         cls_id = int(box.cls[0])
            #         label = self.yolo_model.names[cls_id]
            #         # 编号统计
            #         label_count.setdefault(label, 0)
            #         label_count[label] += 1
            #         label_idx = label_count[label]
            #         label_with_idx = f"{label}{label_idx}"
            #         rospy.loginfo(f"目标 {i}: 类别={label}, 置信度={conf:.3f}, 坐标=({x1},{y1},{x2},{y2})")
            #         cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(yolo_vis, f'{label_with_idx} {conf:.2f}', (x1, y1 - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #         # 检测框中心点三维坐标（与角点一致的投影和坐标系转换）
            #         cx = int((x1 + x2) / 2)
            #         cy = int((y1 + y2) / 2)
            #         if 0 <= cx < depth.shape[1] and 0 <= cy < depth.shape[0]:
            #             z = depth[cy, cx] / self.depth_scale
            #             if z > 0:
            #                 x_cam = (cx - self.K[0, 2]) * z / self.K[0, 0]
            #                 y_cam = (cy - self.K[1, 2]) * z / self.K[1, 1]
            #                 pt3d_cam = [x_cam, y_cam, z]
            #                 pt3d_converted = self.convert_coordinate_system([pt3d_cam])[0]
            #                 det_centers_3d.append((label_with_idx, pt3d_converted[0], pt3d_converted[1], pt3d_converted[2], conf))
            #                 rospy.loginfo(f"目标 {i} 检测框中心点像素: ({cx},{cy}), 三维坐标: ({pt3d_converted[0]:.3f}, {pt3d_converted[1]:.3f}, {pt3d_converted[2]:.3f})")
            #             else:
            #                 rospy.loginfo(f"目标 {i} 检测框中心点像素: ({cx},{cy}), 深度值无效")
            #         else:
            #             rospy.loginfo(f"目标 {i} 检测框中心点像素: ({cx},{cy}) 超出图像范围")

            #     # 可视化和保存
            #     # 左上角显示所有三维坐标
            #     for idx, (label_with_idx, x, y, z, conf) in enumerate(det_centers_3d):
            #         text = f"{label_with_idx} ({conf:.2f}): [{x:.2f}, {y:.2f}, {z:.2f}]"
            #         cv2.putText(yolo_vis, text, (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            #     rospy.loginfo("YOLO检测结果可视化显示中...")
            #     cv2.imshow("YOLO Detection", yolo_vis)
            #     cv2.waitKey(1)
            #     if self.save_images:
            #         det_result_dir = os.path.join(self.base_output_dir, "det_result/")
            #         os.makedirs(det_result_dir, exist_ok=True)
            #         det_filename = os.path.join(det_result_dir, f"{int(time.time())}_yolo_result.png")
            #         cv2.imwrite(det_filename, yolo_vis)
            #         rospy.loginfo(f"保存YOLO检测结果图像: {det_filename}")
            # else:
            #     rospy.loginfo("未检测到目标，跳过YOLO结果保存和可视化")
            #     return

            # 生成实时分割掩码
            mask = self.generate_mask(rgb)
            if mask.sum() == 0:
                rospy.logwarn("No target detected in current frame")
                return


            #========== 平面拟合测试变量保存 ===================================================
            # test_result_dir = os.path.join(self.base_output_dir, "test_result/")
            # os.makedirs(test_result_dir, exist_ok=True)
            # depth_filename = os.path.join(test_result_dir, f"{int(time.time())}_depth.png")
            # mask_path = os.path.join(test_result_dir, f"{int(time.time())}_mask.png")
            # cv2.imwrite(depth_filename, depth)  # depth是16位单通道，直接保存保持深度信息
            # cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))  # mask二值图，转成0-255保存

            # 执行处理流程
            plane_normal, corners, corners_camera = self.process_images(rgb, depth, mask)
            
            # 检查处理结果
            if plane_normal is None or len(corners) == 0:
                rospy.logwarn("平面检测失败，跳过当前帧")
                return

            # 显示合并的可视化图像（分割结果+角点结果）
            if self.enable_visualization:
                rospy.loginfo("开始创建合并可视化...")
                # 可视化需要使用相机坐标系进行投影，但显示转换后的坐标
                self.show_combined_visualization(rgb, mask, corners_camera, corners, depth)
                rospy.loginfo("合并可视化完成")
            else:
                rospy.loginfo("可视化已禁用，跳过窗口显示")
            
            # 可选：保存图像
            if self.save_images:
                # 保存时也需要使用相机坐标系进行投影，显示转换后坐标
                self.save_corner_image(rgb, corners_camera, corners, depth)
            

            # 将平移后的角点转换为目标坐标系（含单位转换，输出单位：厘米）
            # corners_shifted_converted = self.convert_coordinate_system(corners_camera)

            #========== 角点到原点距离检查 ===================================================
            if not self.check_corners_distance(corners):
                rospy.logwarn("角点距离检查未通过，跳过当前帧的消息发布")
                return

            #========== 创建并发布结果消息 - 优化输出排版（无时间戳） =============================================
            print("=" * 60)
            print("【平面检测结果】")
            print("=" * 60)
            print(f"平面法向量: [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
            print("-" * 60)
            print("四个角点坐标 (坐标系: x前, y右, z上) - 基于图像位置排序:")
            corner_labels = [
                "P1-左下(左近)", 
                "P2-右下(右近)", 
                "P3-右上(右远)", 
                "P4-左上(左远)"
            ]
            
            for i, corner in enumerate(corners):
                print(f"  {corner_labels[i]}: [{corner[0]:.1f}, {corner[1]:.1f}, {corner[2]:.1f}] mm")
            print("=" * 60)


            msg = PlaneCorners()
            msg.normal = Vector3(*plane_normal.tolist())
            msg.corners = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners]
            
            # 添加发布调试信息
            rospy.loginfo("=" * 40)
            rospy.loginfo("[ROS发布] 正在发布PlaneCorners消息到话题 /result")
            rospy.loginfo(f"[ROS发布] 法向量: [{msg.normal.x:.3f}, {msg.normal.y:.3f}, {msg.normal.z:.3f}]")
            rospy.loginfo(f"[ROS发布] 角点数量: {len(msg.corners)}")
            for i, corner in enumerate(msg.corners):
                rospy.loginfo(f"[ROS发布] 角点{i+1}: [{corner.x:.1f}, {corner.y:.1f}, {corner.z:.1f}] mm")
            rospy.loginfo(f"[ROS发布] 发布者连接数: {self.result_pub.get_num_connections()}")
            
            self.result_pub.publish(msg)
            rospy.loginfo("[ROS发布] 消息已发布!")
            rospy.loginfo("=" * 40)

            rospy.loginfo(f"[图像处理完成] 耗时: {time.time() - start_time:.3f} 秒")

        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")


    def check_corners_distance(self, corners):
        min_corner_distance = 700.0    # mm
        max_corner_distance = 1500.0   # mm
        jump_threshold = 200.0         # mm

        corner_labels = [
            "（左）近距点", 
            "（右）近距点", 
            "（右）远距点", 
            "（左）远距点"
        ]
        
        corner_distances = []
        abnormal_found = False

        for i, corner in enumerate(corners):
            dist = np.linalg.norm(corner)
            corner_distances.append(dist)
            if dist < min_corner_distance or dist > max_corner_distance:
                rospy.logwarn(f"角点 {corner_labels[i]} 到原点距离异常: {dist:.1f} mm")
                abnormal_found = True

        if hasattr(self, 'last_corner_distances'):
            for i, (curr_d, prev_d) in enumerate(zip(corner_distances, self.last_corner_distances)):
                delta = abs(curr_d - prev_d)
                if delta > jump_threshold:
                    rospy.logwarn(f"角点 {corner_labels[i]} 距离突变: 上一帧 {prev_d:.1f} mm -> 当前帧 {curr_d:.1f} mm，变化 {delta:.1f} mm")
                    abnormal_found = True

        self.last_corner_distances = corner_distances.copy()
        return not abnormal_found  # True 表示"没有异常"


    def show_combined_visualization(self, rgb, mask, corners_3d_camera, corners_3d_converted, depth):
        """显示合并的可视化图像：分割结果+角点结果"""
        try:
            # 创建分割可视化
            mask_vis = self.create_mask_visualization(rgb, mask)
            
            # 创建角点可视化 (传入相机坐标系数据用于投影，转换后数据用于显示)  
            corner_vis = self.create_corner_visualization(rgb, corners_3d_camera, corners_3d_converted)
            
            # 检查图像是否有效
            if mask_vis is None or corner_vis is None:
                rospy.logerr("创建的可视化图像无效")
                return
            
            # 确保两个图像大小一致
            h, w = rgb.shape[:2]
            mask_vis = cv2.resize(mask_vis, (w, h))
            corner_vis = cv2.resize(corner_vis, (w, h))
            
            # 水平拼接两个图像
            combined = np.hstack((mask_vis, corner_vis))
            
            # 添加标题
            title_height = 40
            title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_img, "YOLO Segmentation", (20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(title_img, "Corner Detection", (w + 20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 垂直拼接标题和图像
            final_image = np.vstack((title_img, combined))
            
            # 显示合并的窗口
            cv2.imshow("Plane Detection - Segmentation and Corners", final_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"显示合并可视化失败: {str(e)}")
            import traceback
            rospy.logerr(f"详细错误信息: {traceback.format_exc()}")


    def save_corner_image(self, rgb, corners_3d_camera, corners_3d_converted, depth):
        """保存带有四个角点可视化的图像"""
        try:
            vis_image = self.create_corner_visualization(rgb, corners_3d_camera, corners_3d_converted)
            
            # 保存图像
            timestamp = str(int(time.time()))
            filename = os.path.join(self.photo_dir, f"{timestamp}_corners_result.png")
            cv2.imwrite(filename, vis_image)
            rospy.loginfo(f"保存角点可视化图像: {filename}")
            
        except Exception as e:
            rospy.logerr(f"保存角点可视化失败: {str(e)}")

    def create_corner_visualization(self, rgb, corners_3d_camera, corners_3d_converted):
        """创建带有四个角点可视化的图像"""
        try:
            # 输入检查
            if rgb is None or len(rgb.shape) != 3:
                rospy.logerr("输入RGB图像无效")
                return None
            
            if corners_3d_camera is None or len(corners_3d_camera) != 4:
                rospy.logerr(f"相机坐标角点数据无效: {corners_3d_camera}")
                return None
                
            if corners_3d_converted is None or len(corners_3d_converted) != 4:
                rospy.logerr(f"转换后角点数据无效: {corners_3d_converted}")
                return None
            
            vis_image = rgb.copy()
            # 将3D角点投影到2D图像坐标（使用相机坐标系数据）
            corner_pixels = []
            for i, corner_3d in enumerate(corners_3d_camera):
                x_3d, y_3d, z_3d = corner_3d
                
                if z_3d <= 0:
                    # 使用默认位置
                    u, v = 50 + i * 100, 50
                else:
                    # 3D到2D投影
                    u = int(x_3d * self.K[0, 0] / z_3d + self.K[0, 2])
                    v = int(y_3d * self.K[1, 1] / z_3d + self.K[1, 2])
                
                # 确保坐标在图像范围内
                u = max(0, min(rgb.shape[1] - 1, u))
                v = max(0, min(rgb.shape[0] - 1, v))
                
                corner_pixels.append((u, v))
            
            # 绘制角点和连线
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红绿蓝黄
            
            # 首先绘制连接线（在角点下方）
            for i in range(4):
                pt1 = corner_pixels[i]
                pt2 = corner_pixels[(i + 1) % 4]
                cv2.line(vis_image, pt1, pt2, (0, 255, 255), 4)  # 加粗青色连线
            
            # 绘制角点（在连线上方）
            corner_labels = ["LD", "RD", "RU", "LU"]  # 左下, 右下, 右上, 左上
            corner_descriptions = ["左下(左近)", "右下(右近)", "右上(右远)", "左上(左远)"]
            for i, (u, v) in enumerate(corner_pixels):
                # 绘制外圈（白色）
                cv2.circle(vis_image, (u, v), 15, (255, 255, 255), -1)
                # 绘制内圈（彩色）
                cv2.circle(vis_image, (u, v), 10, colors[i], -1)
                # 绘制标号
                cv2.putText(vis_image, f"{i+1}", (u-8, v+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # 绘制角点位置标签
                cv2.putText(vis_image, f"P{i+1}-{corner_labels[i]}", (u+18, v-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_image, f"P{i+1}-{corner_labels[i]}", (u+18, v-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            
            # 添加背景板使文字更清晰
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
            
            # 添加信息文本（显示转换后的坐标，单位为毫米）
            info_text = [
                "Plane Detection - 4 Corners (P1:LD->P2:RD->P3:RU->P4:LU)",
                f"P1-LeftDown: ({corners_3d_converted[0][0]:.1f}, {corners_3d_converted[0][1]:.1f}, {corners_3d_converted[0][2]:.1f})mm",
                f"P2-RightDown: ({corners_3d_converted[1][0]:.1f}, {corners_3d_converted[1][1]:.1f}, {corners_3d_converted[1][2]:.1f})mm",
                f"P3-RightUp: ({corners_3d_converted[2][0]:.1f}, {corners_3d_converted[2][1]:.1f}, {corners_3d_converted[2][2]:.1f})mm",
                f"P4-LeftUp: ({corners_3d_converted[3][0]:.1f}, {corners_3d_converted[3][1]:.1f}, {corners_3d_converted[3][2]:.1f})mm"
            ]


            # 在图像上绘制信息
            for i, text in enumerate(info_text):
                y_pos = 25 + i * 20
                # 绘制文字（白色，更清晰）
                cv2.putText(vis_image, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 计算并显示平面尺寸（使用转换后的坐标，单位为毫米）
            p1, p2, p3, p4 = corners_3d_converted
            width1 = np.linalg.norm(np.array(p2) - np.array(p1))
            width2 = np.linalg.norm(np.array(p4) - np.array(p3))
            height1 = np.linalg.norm(np.array(p4) - np.array(p1))
            height2 = np.linalg.norm(np.array(p3) - np.array(p2))
            
            avg_width = (width1 + width2) / 2
            avg_height = (height1 + height2) / 2
            
            # 添加尺寸信息
            size_text = f"Size: {avg_width:.1f}mm x {avg_height:.1f}mm"
            cv2.putText(vis_image, size_text, (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 角点可视化图像创建成功
            return vis_image
            
        except Exception as e:
            rospy.logerr(f"创建角点可视化失败: {str(e)}")
            import traceback
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            return rgb.copy()  # 返回原图

    def create_mask_visualization(self, rgb, mask):
        """创建分割掩码可视化图像"""
        try:
            vis_rgb = rgb.copy()
            
            # 找到轮廓并绘制
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_rgb, contours, -1, (0, 255, 0), 2)
            
            # 创建彩色掩码叠加
            colored_mask = np.zeros_like(rgb)
            colored_mask[mask == 1] = [0, 255, 0]  # 绿色
            overlay = cv2.addWeighted(vis_rgb, 0.7, colored_mask, 0.3, 0)
            
            # 添加信息
            mask_area = np.sum(mask)
            info_text = f"Mask area: {mask_area} pixels"
            cv2.putText(overlay, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return overlay
            
        except Exception as e:
            rospy.logerr(f"创建分割可视化失败: {str(e)}")
            return rgb.copy()

    def generate_mask(self, rgb):
        """使用YOLO生成实时分割掩码（改进版：置信度筛选+形态学处理）"""
        rospy.loginfo("进入 generate_mask")
        results = self.model.predict(
            rgb,
            imgsz=640,
            half=True,
            verbose=False,
            conf=self.confidence_threshold,
            # device='cuda'  # 强制使用GPU
        )
        
        best_mask = None
        best_confidence = 0
        
        for result in results:
            if result.masks is not None and len(result.masks.data) > 0:
                # 遍历所有检测结果，选择置信度最高的
                for i, mask_data in enumerate(result.masks.data):
                    if result.boxes is not None and i < len(result.boxes.conf):
                        confidence = float(result.boxes.conf[i])
                        rospy.loginfo(f"检测到目标 {i}: 置信度 {confidence:.3f}")
                        
                        if confidence > best_confidence and confidence > self.confidence_threshold:  # 置信度阈值
                            best_mask = mask_data.cpu().numpy()
                            best_confidence = confidence
                
        if best_mask is not None:
            # 二值化处理
            mask_bin = (best_mask > 0.5).astype(np.uint8)
            
            # 形态学操作：去除小噪点和填充空洞
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)  # 填充空洞
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)   # 去除噪点
            
            # 可视化并保存分割结果
            vis_rgb = rgb.copy()
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_rgb, contours, -1, (0, 255, 0), 2)
            colored_mask = np.zeros_like(rgb)
            colored_mask[mask_bin == 1] = [0, 255, 0]
            overlay = cv2.addWeighted(vis_rgb, 0.7, colored_mask, 0.3, 0)
            
            # 添加置信度信息
            cv2.putText(overlay, f"Confidence: {best_confidence:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            timestamp = str(int(time.time()))
            filename = os.path.join(self.photo_dir, f"{timestamp}_mask_overlay.png")
            
            if self.save_images:
                cv2.imwrite(filename, overlay)
                rospy.loginfo(f"保存分割可视化图像: {filename}, 置信度: {best_confidence:.3f}")
            else:
                rospy.loginfo(f"分割完成，置信度: {best_confidence:.3f} (图像保存已禁用)")
            
            # 分割结果将在合并窗口中显示

            return mask_bin
    
        # 如果没有掩码，返回全0掩码
        rospy.logwarn("未检测到足够置信度的目标")
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    def process_images(self, rgb, depth, mask):
        # === 掩码预处理，去除零散的小范围掩码===================================
        # 预处理掩码 - 更灵活的面积阈值
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask)
        
        # 动态计算面积阈值 (图像总面积的0.1%)
        total_area = mask.shape[0] * mask.shape[1]
        min_area = max(500, int(total_area * 0.001))
        rospy.loginfo(f"轮廓面积阈值: {min_area} 像素")
        
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask_clean, [contour], -1, 1, cv2.FILLED)
                valid_contours += 1
                rospy.loginfo(f"有效轮廓面积: {area:.0f} 像素")
        
        mask = mask_clean
        rospy.loginfo(f"保留 {valid_contours} 个有效轮廓")

        # 生成点云
        ys, xs = np.where(mask == 1)
        rospy.loginfo(f"掩码有效像素数量: {len(xs)}")
        
        if len(xs) == 0:
            raise ValueError("Valid mask area is empty")

        zs = depth[ys, xs].astype(np.float32) / self.depth_scale
        rospy.loginfo(f"深度值统计: min={zs.min():.3f}m, max={zs.max():.3f}m, 非零数量={np.sum(zs > 0)}")
        

        # ===  根据深度范围和集群去滤波 ===================================
        # 自适应深度范围调整
        valid_depths = zs[zs > 0]
        if len(valid_depths) > 0:
            depth_mean = np.mean(valid_depths)
            depth_std = np.std(valid_depths)
            
            # 使用3-sigma规则设置动态范围
            depth_min = max(0.1, depth_mean - 2*depth_std)
            depth_max = min(3.0, depth_mean + 2*depth_std)
            rospy.loginfo(f"自适应深度范围: {depth_min:.3f}m - {depth_max:.3f}m (均值: {depth_mean:.3f}m)")
        else:
            depth_min, depth_max = 0.2, 1.5
            rospy.logwarn("没有有效深度值，使用默认范围: 0.2-1.5m")
        
        depth_valid = (zs > depth_min) & (zs < depth_max)
        rospy.loginfo(f"深度范围过滤后: {np.sum(depth_valid)}个点")
        
        xs = xs[depth_valid]
        ys = ys[depth_valid]
        zs = zs[depth_valid]

        if len(xs) < 10:
            rospy.logerr(f"有效点数太少: {len(xs)}个，无法进行可靠的平面拟合")
            raise ValueError(f"Insufficient valid points: {len(xs)}")

        # 坐标转换
        xs3d = (xs - self.K[0, 2]) * zs / self.K[0, 0]
        ys3d = (ys - self.K[1, 2]) * zs / self.K[1, 1]
        points = np.stack((xs3d, ys3d, zs), axis=-1)
        
        # 去除异常值
        valid = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        points = points[valid]
        rospy.loginfo(f"坐标转换后有效点数: {len(points)}")

        if len(points) < 10:
            rospy.logerr(f"坐标转换后点数不足: {len(points)}个")
            raise ValueError(f"Insufficient points after coordinate conversion: {len(points)}")
        
        # 转换为Open3D点云格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # self.debug_show_pointcloud(points)  # 调试显示点云

        # 自适应去噪参数
        nb_neighbors = min(20, max(5, len(points) // 10))
        pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=1.5)
        points = np.asarray(pcd.points)
        # self.debug_show_pointcloud(points)  # 调试显示点云

        rospy.loginfo(f"统计滤波后点云: {len(points)}个点 (去除了{len(inlier_indices) - len(points)}个离群点)")

        if len(points) < 3:
            rospy.logerr(f"滤波后点数不足进行平面拟合: {len(points)}个")
            raise ValueError(f"Insufficient points after filtering: {len(points)}")
        


        # === 第一次拟合平面 ===================================
        distance_threshold = 0.005  # 5mm容差，更严格
        ransac_n = 3
        num_iterations = 1000  # 增加迭代次数提高精度
        
        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        rospy.loginfo(f"RANSAC平面拟合: {len(inliers)}个内点 / {len(points)}个总点 (内点率: {len(inliers)/len(points)*100:.1f}%)")
        
        if len(inliers) < len(points) * 0.3:  # 至少30%的点应该在平面上
            rospy.logwarn(f"平面拟合质量较差，内点率只有 {len(inliers)/len(points)*100:.1f}%")
        
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)
        rospy.loginfo(f"第一次拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")


        # === 二次滤波 ===================================
        distances = np.abs((points @ plane_normal) + d)
        threshold = 0.01  # 1cm
        filtered_points = points[distances < threshold]
        rospy.loginfo(f"过滤平面外点: 距离 < 1cm 点数为 {len(filtered_points)} / {len(points)}")

        if len(filtered_points) < 10:
            rospy.logwarn("过滤后剩余点数过少，拟合或mask可能有误")

        points = filtered_points  # 用更精确的点集替代



        # ==== 二次拟合平面 ==================================
        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        rospy.loginfo(f"RANSAC平面拟合: {len(inliers)}个内点 / {len(points)}个总点 (内点率: {len(inliers)/len(points)*100:.1f}%)")
        
        if len(inliers) < len(points) * 0.3:  # 至少30%的点应该在平面上
            rospy.logwarn(f"平面拟合质量较差，内点率只有 {len(inliers)/len(points)*100:.1f}%")
        
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)
        rospy.loginfo(f"第二次拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")



        # ==== 将点云投影到拟合的平面上 =====================================
        # 点云投影
        t = -(points @ plane_normal + d) / (a**2 + b**2 + c**2)
        projected_points = points + t[:, np.newaxis] * plane_normal
        center = np.mean(projected_points, axis=0)
        # self.debug_show_pointcloud(projected_points)  # 调试显示点云

        # 计算外接矩形
        up = np.array([0, 0, 1])
        if np.allclose(plane_normal, up):
            up = np.array([0, 1, 0])
        x_axis = np.cross(up, plane_normal)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(plane_normal, x_axis)

        points_2d = np.stack([
            (projected_points - center) @ x_axis,
            (projected_points - center) @ y_axis
        ], axis=1).astype(np.float32)

        rect = cv2.minAreaRect(points_2d)
        box_2d = cv2.boxPoints(rect)
        
        # 获取未排序的角点（相机坐标系）
        corner_points_3d_unsorted = [center + x*x_axis + y*y_axis for x, y in box_2d]
        
        
        # 计算矩形尺寸
        width, height = rect[1]
        rospy.loginfo(f"检测到的平面尺寸: {width:.3f}m × {height:.3f}m")

        # 绘制并保存结果
        colors = np.zeros_like(points)
        colors[:, 1] = 1.0  # Green

        # ========== 修复：首先对相机坐标系角点进行排序 ==========
        # 在相机坐标系中先进行排序，确保投影和文字一致
        corner_points_3d_camera_sorted = self.sort_corners_in_camera_coordinates(corner_points_3d_unsorted)
        
        # 然后对排序后的相机坐标进行坐标系转换
        corner_points_3d_converted = self.convert_coordinate_system(corner_points_3d_camera_sorted)
        
        # 用于点云可视化的相机坐标系数据（保持排序一致性）
        corner_points_3d_camera_for_display = corner_points_3d_camera_sorted

        # 添加角点（红色）和边框线段（蓝色） - 使用排序后的相机坐标系进行点云可视化
        red = [1.0, 0.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        corner_points = np.array(corner_points_3d_camera_sorted)  # 使用排序后的相机坐标系数据

        # 添加角点为红色
        points_with_visuals = np.vstack([points, corner_points])
        colors_with_visuals = np.vstack([colors, np.tile(red, (4, 1))])

        # 假设你想给 projected_points 设置灰色，比如 [0.5, 0.5, 0.5]
        proj_color = [0.5, 0.5, 0.5]

        # points_with_visuals、colors_with_visuals 已经包含 points 和角点、连线，现在添加投影点
        points_with_visuals = np.vstack([points_with_visuals, projected_points])
        colors_with_visuals = np.vstack([colors_with_visuals, np.tile(proj_color, (len(projected_points), 1))])

        # 添加边框线条为蓝色（连接每两个相邻角点）
        for i in range(4):
            p1 = corner_points[i]
            p2 = corner_points[(i + 1) % 4]
            line = np.linspace(p1, p2, num=50)  # 每条边插值 50 个点
            points_with_visuals = np.vstack([points_with_visuals, line])
            colors_with_visuals = np.vstack([colors_with_visuals, np.tile(blue, (50, 1))])

        # 创建彩色点云对象
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points_with_visuals)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors_with_visuals)

        # 保存点云和掩码图
        timestamp = str(int(time.time()))
        
        if self.save_pointclouds:
            pcd_path = os.path.join(self.pcd_dir, f"{timestamp}.pcd")
            o3d.io.write_point_cloud(pcd_path, pcd_vis)
            rospy.loginfo(f"保存点云: {pcd_path}")
        else:
            rospy.loginfo("点云处理完成 (点云保存已禁用)")


        # 处理法向量转换
        plane_normal_shift = self.convert_coordinate_system(plane_normal)
        origin_shift = self.convert_coordinate_system(np.array([0,0,0]))  # 平移向量转换

        # 确保先转换为 numpy array
        p0 = np.array(origin_shift, dtype=np.float32)
        p1 = np.array(plane_normal_shift, dtype=np.float32)

        direction = p1 - p0
        norm = np.linalg.norm(direction)

        if norm == 0:
            rospy.logwarn("两个点重合，无法计算法向量")
            return [0, 0, 1]  # 默认值

        plane_normal_shift = direction / norm

        return plane_normal_shift, corner_points_3d_converted, corner_points_3d_camera_for_display


    def convert_coordinate_system(self, corners_camera):
        """
        使用旋转矩阵将相机坐标系转换为目标坐标系，并应用单位转换。
        内部先对角点进行原点平移（单位：米），再进行旋转变换和单位转换。
        支持输入为单个3D点（list/np.array长度为3）或多个点的列表/数组。
        """
        try:
            # 定义相机 → 目标坐标系的旋转矩阵
            if self.coordinate_system == 'right_hand':
                R = np.array([
                    [0, 0, 1],   # x_new = z_cam
                    [1, 0, 0],   # y_new = x_cam
                    [0, 1, 0],  # z_new = y_cam
                ])
            else:
                R = np.identity(3)

            # 原点平移向量（单位：米）
            origin_shift = np.array([0.043, 0.050, 0.0])  # 右4.3mm，下5mm

            # 转换输入为numpy数组，方便批量操作
            corners_np = np.array(corners_camera, dtype=np.float32)

            # 对单点情况
            if corners_np.shape == (3,):
                # 原点平移
                shifted = corners_np - origin_shift
                # 旋转矩阵变换
                converted = R @ shifted.reshape(3, 1)
                # 单位转换（比如米转厘米，self.unit_scale一般是100）
                converted = (converted * self.unit_scale).flatten()
                return converted.tolist()

            # 对多个点情况 (Nx3)
            elif corners_np.ndim == 2 and corners_np.shape[1] == 3:
                # 原点平移，广播减法
                shifted = corners_np - origin_shift
                # 旋转矩阵变换
                converted = (R @ shifted.T).T  # 变换后仍为 Nx3
                # 单位转换
                converted *= self.unit_scale
                return converted.tolist()

            else:
                rospy.logwarn("输入格式不支持，返回原始值")
                return corners_camera

        except Exception as e:
            rospy.logerr(f"坐标系转换失败: {str(e)}")
            return corners_camera



    def sort_corners_in_camera_coordinates(self, corners_3d):
        """
        对四个角点进行排序，使用基于图像投影位置的策略：
        P1: 左下角(左近) - 图像左下
        P2: 右下角(右近) - 图像右下  
        P3: 右上角(右远) - 图像右上
        P4: 左上角(左远) - 图像左上
        
        ⚠️ 使用图像投影坐标进行排序，确保可视化一致性
        单位：毫米(mm)
        """
        try:
            # 转换为numpy数组便于计算
            corners_array = np.array(corners_3d)
            
                         # 计算质心
            center = np.mean(corners_array, axis=0)
            rospy.loginfo(f"【相机坐标系】角点质心: [{center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}]mm")
            
            # 将3D角点投影到2D图像坐标，用于排序
            corners_with_projection = []
            for i, corner_3d in enumerate(corners_array):
                x_3d, y_3d, z_3d = corner_3d
                
                if z_3d > 0:  # 确保深度有效
                    u = x_3d * self.K[0, 0] / z_3d + self.K[0, 2]  # 图像x坐标
                    v = y_3d * self.K[1, 1] / z_3d + self.K[1, 2]  # 图像y坐标
                else:
                    # 如果深度无效，使用默认位置
                    u, v = 320, 240
                
                dist_3d = np.linalg.norm(corner_3d) * 1000  # 转换为mm
                corners_with_projection.append({
                    'corner_3d': corner_3d,
                    'u': u, 'v': v,
                    'dist_3d': dist_3d,
                    'original_idx': i
                })
                
                rospy.loginfo(f"原始角点{i}: 3D[{x_3d*1000:.1f}, {y_3d*1000:.1f}, {z_3d*1000:.1f}]mm → 2D投影({u:.1f}, {v:.1f})px, 距离={dist_3d:.1f}mm")
            
                         # 使用图像坐标进行排序（更直观和稳定）
            # 1. 按V坐标（图像Y）分上下两组
            corners_with_projection.sort(key=lambda x: x['v'])
            top_corners = corners_with_projection[:2]    # 图像上方（V小）
            bottom_corners = corners_with_projection[2:]  # 图像下方（V大）
            
            # 2. 在每组内按U坐标（图像X）排序
            top_corners.sort(key=lambda x: x['u'])        # 上方：左到右
            bottom_corners.sort(key=lambda x: x['u'])     # 下方：左到右
            
            # 3. 按照用户要求的顺序分配：P1左下(左近)→P2右下(右近)→P3右上(右远)→P4左上(左远)
            sorted_corner_data = [
                bottom_corners[0],   # P1: 左下 (左近)
                bottom_corners[1],   # P2: 右下 (右近)
                top_corners[1],      # P3: 右上 (右远)
                top_corners[0]       # P4: 左上 (左远)
            ]
            
            # 提取排序后的3D坐标
            sorted_corners = [item['corner_3d'] for item in sorted_corner_data]
            
            # 输出详细调试信息
            corner_names = ["左下(左近)", "右下(右近)", "右上(右远)", "左上(左远)"]
            rospy.loginfo(f"【相机坐标系】排序策略: 基于图像投影位置排序")
            rospy.loginfo("=" * 50)
            for i, (data, name) in enumerate(zip(sorted_corner_data, corner_names)):
                corner = data['corner_3d']
                u, v = data['u'], data['v']
                dist = data['dist_3d']
                orig_idx = data['original_idx']
                rospy.loginfo(f"P{i+1} {name}: 原始索引{orig_idx} → 图像位置({u:.0f},{v:.0f})px, 3D距离={dist:.1f}mm")
                rospy.loginfo(f"     3D坐标: [{corner[0]*1000:.1f}, {corner[1]*1000:.1f}, {corner[2]*1000:.1f}]mm")
            rospy.loginfo("=" * 50)
            
            return sorted_corners
            
        except Exception as e:
            rospy.logerr(f"角点排序失败: {str(e)}")
            import traceback
            rospy.logerr(f"详细错误: {traceback.format_exc()}")
            # 如果排序失败，返回原始顺序
            return corners_3d

if __name__ == '__main__':
    try:
        detector = PlaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
