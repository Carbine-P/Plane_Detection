import cv2
import numpy as np
import open3d as o3d
import rospy
import time
import os

class PlaneDetector:
    def __init__(self):
        # 示例相机内参fx, fy, cx, cy，需换成实际参数
        self.K = np.array([[615.0, 0, 320.0],
                           [0, 615.0, 240.0],
                           [0, 0, 1]])
        self.depth_scale = 1000.0  # 深度图单位转换，mm转m
        self.save_pointclouds = False
        self.pcd_dir = "./pcd_output"
        os.makedirs(self.pcd_dir, exist_ok=True)

    def sort_corners(self, corners):
        # 简单实现：根据xy排序，返回左上、右上、右下、左下
        corners = np.array(corners)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()

        ordered = np.zeros((4, 3))
        ordered[0] = corners[np.argmin(s)]  # 左上
        ordered[2] = corners[np.argmax(s)]  # 右下
        ordered[1] = corners[np.argmin(diff)]  # 右上
        ordered[3] = corners[np.argmax(diff)]  # 左下
        return ordered

    def debug_show_pointcloud(self,points_np):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axis], window_name="调试点云")

    # 这里粘贴你之前的 process_images 函数代码
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
        num_iterations = 2000  # 增加迭代次数提高精度
        
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
        t = -(points @ plane_normal + d) / (a**2 + b**2 + c**2)
        projected_points = points + t[:, np.newaxis] * plane_normal
        # self.debug_show_pointcloud(projected_points)  # 调试显示点云

        center = np.mean(projected_points, axis=0)

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

        for i, pt in enumerate(box_2d):
            rospy.loginfo(f"box_2d[{i}] = {pt}")
        
        # 对角点进行排序，确保固定顺序：左上 -> 右上 -> 右下 -> 左下
        corner_points_3d_unsorted = [center + x*x_axis + y*y_axis for x, y in box_2d]
        # corner_points_3d_camera = self.sort_corners(corner_points_3d_unsorted)
        corner_points_3d_camera = corner_points_3d_unsorted
        
        
        # 计算矩形尺寸
        width, height = rect[1]
        rospy.loginfo(f"检测到的平面尺寸: {width:.3f}m × {height:.3f}m")


        # 绘制并保存结果
        colors = np.zeros_like(points)
        colors[:, 1] = 1.0  # Green

        # 添加角点（红色）和边框线段（蓝色） - 使用原始相机坐标系进行点云可视化
        red = [1.0, 0.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        corner_points = np.array(corner_points_3d_camera)  # 使用相机坐标系数据进行点云可视化

        print("角点 (相机坐标系):", corner_points)

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

        # 可视化彩色点云
        o3d.visualization.draw_geometries([pcd_vis], window_name="Plane Detection Result")

        # 保存点云和掩码图
        timestamp = str(int(time.time()))
        
        if self.save_pointclouds:
            pcd_path = os.path.join(self.pcd_dir, f"{timestamp}.pcd")
            o3d.io.write_point_cloud(pcd_path, pcd_vis)
            rospy.loginfo(f"保存点云: {pcd_path}")
        else:
            rospy.loginfo("点云处理完成 (点云保存已禁用)")


        # # 转换坐标系并应用单位转换
        # corner_points_3d = self.convert_coordinate_system(corner_points_3d_camera)
        # R = np.array([
        #     [0, 0, 1],
        #     [1, 0, 0],
        #     [0, 1, 0],
        # ])
        # plane_normal_shift = R @ plane_normal
        # plane_normal_shift /= np.linalg.norm(plane_normal_shift)

        return plane_normal, corner_points_3d, corner_points_3d_camera



if __name__ == "__main__":
    rospy.init_node("plane_detector_test_node", anonymous=True)

    # 创建检测器实例
    detector = PlaneDetector()

    # 读取测试数据（请替换成你自己测试的路径）
    rgb = cv2.imread("/home/orin/PHT/workspace/src/plane_detection/output/photo/1750745822_corners_result.png")          # BGR格式
    depth = cv2.imread("/home/orin/PHT/workspace/src/plane_detection/output/test_result/1750745822_depth.png", cv2.IMREAD_UNCHANGED)  # 深度图，单位通常为mm或16位整数
    mask = cv2.imread("/home/orin/PHT/workspace/src/plane_detection/output/test_result/1750745822_mask.png", cv2.IMREAD_GRAYSCALE)   # 掩码，单通道，0和255

    if rgb is None or depth is None or mask is None:
        print("Error: 请确认 test_rgb.png, test_depth.png, test_mask.png 文件存在！")
        exit(-1)

    # 二值化掩码（确保是0/1）
    _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    try:
        plane_normal_shift, corner_points_3d, corner_points_3d_camera = detector.process_images(rgb, depth, mask_binary)

        print("平面法向量 (转换后坐标系):", plane_normal_shift)
        print("角点 (转换后坐标系):")
        print(corner_points_3d)
        print("角点 (相机坐标系):")
        print(corner_points_3d_camera)

        # 可视化（Open3D）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(corner_points_3d_camera)
        colors = [[1, 0, 0] for _ in range(len(corner_points_3d_camera))]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name="Plane Corners (Camera Coord)")

    except Exception as e:
        print(f"处理错误: {e}")
