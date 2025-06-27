#!/home/ecnu/anaconda3/envs/yolov11/bin/python
import rospy
import cv2
import numpy as np
import json
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO
import os

class ObjectDetector3D:
    def __init__(self):
        rospy.init_node('object_detector_3d', anonymous=True)

        # 加载相机参数
        camera_json_path = rospy.get_param('~camera_json_path', '/home/ecnu/PHT/PosDet/data/camera.json')
        with open(camera_json_path, "r") as f:
            cam_data = json.load(f)
        self.K = np.array(cam_data["cam_K"]).reshape(3, 3)
        self.depth_scale = cam_data["depth_scale"]
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

        # 加载检测模型
        model_path = rospy.get_param('~model_path', '/home/ecnu/PHT/yolo/runs/detect/train5/weights/best.pt')
        self.model = YOLO(model_path)
        rospy.loginfo(f"Loaded detection model from: {model_path}")

        # Bridge
        self.bridge = CvBridge()

        # 发布检测目标中心点
        self.center_pub = rospy.Publisher('/object_centers', PointStamped, queue_size=10)

        # 同步 RGB + 深度订阅
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)


    def image_callback(self, rgb_msg, depth_msg):
        try:
            # 转换图像
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

            # 检测
            results = self.model(rgb)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # 中心像素坐标
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                if v >= depth.shape[0] or u >= depth.shape[1]:
                    rospy.logwarn("检测中心点超出图像边界")
                    continue

                # 深度值转换为米
                z = depth[v, u] / self.depth_scale
                if z <= 0 or z > 5.0:
                    rospy.logwarn(f"无效深度: {z} m")
                    continue

                x_cam = (u - self.cx) * z / self.fx
                y_cam = (v - self.cy) * z / self.fy
                z_cam = z

                rospy.loginfo(f"目标类别: {cls_id}, 置信度: {conf:.2f}")
                rospy.loginfo(f"3D 坐标: X={x_cam:.3f} m, Y={y_cam:.3f} m, Z={z_cam:.3f} m")

                # 发布为 ROS 消息
                point_msg = PointStamped()
                point_msg.header = rgb_msg.header
                point_msg.point.x = x_cam
                point_msg.point.y = y_cam
                point_msg.point.z = z_cam
                self.center_pub.publish(point_msg)

                # 可视化
                cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(rgb, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(
                    rgb,
                    f"({x_cam:.2f},{y_cam:.2f},{z_cam:.2f})",
                    (u, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

            image_path = os.path.join("/home/ecnu/workspace/src/plane_detection/output", 'result.png')
            cv2.imwrite(image_path, rgb)

        except Exception as e:
            rospy.logerr(f"图像处理失败: {str(e)}")


if __name__ == '__main__':
    try:
        detector = ObjectDetector3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
