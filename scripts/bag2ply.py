#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import time
import os
import json

class RGBDPointCloudSaver:
    def __init__(self):
        rospy.init_node('rgbd_pointcloud_saver', anonymous=True)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.last_save_time = time.time()
        self.save_interval = 3  # seconds

        # === Load intrinsics from camera.json ===
        camera_config_path = "/home/ecnu/PHT/PosDet/data/camera.json"
        with open(camera_config_path, 'r') as f:
            cam_data = json.load(f)

        cam_K = cam_data["cam_K"]
        self.depth_scale = cam_data.get("depth_scale", 1000.0)

        # width and height can be hardcoded or added in JSON
        width, height = 640, 480

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(
            width=width,
            height=height,
            fx=cam_K[0],  # fx
            fy=cam_K[4],  # fy
            cx=cam_K[2],  # cx
            cy=cam_K[5]   # cy
        )

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        rospy.loginfo("RGBD PointCloud Saver Initialized.")
        rospy.spin()

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_save()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.try_save()

    def try_save(self):
        if self.rgb_image is None or self.depth_image is None:
            return

        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            timestamp = int(current_time)
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()

            self.save_rgbd_pointcloud(rgb, depth, timestamp)
            self.last_save_time = current_time

    def save_rgbd_pointcloud(self, color_img, depth_img, timestamp):
        rospy.loginfo(f"Saving RGBD and pointcloud at {timestamp}")

        cv2.imwrite(f"color_{timestamp}.png", color_img)
        cv2.imwrite(f"depth_{timestamp}.png", depth_img)

        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image((depth_img.astype(np.float32)) / self.depth_scale)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,  # already scaled
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.intrinsic
        )

        # Flip to align with ROS camera coords
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        ply_filename = f"cloud_{timestamp}.ply"
        o3d.io.write_point_cloud(ply_filename, pcd)
        rospy.loginfo(f"Point cloud saved: {ply_filename}")

if __name__ == '__main__':
    try:
        RGBDPointCloudSaver()
    except rospy.ROSInterruptException:
        pass
