#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetectorNode:
    def __init__(self):
        rospy.init_node('yolo_detector_node')
        self.model = YOLO("/home/ecnu/PHT/yolo/runs/detect/train5/weights/best.pt")
        self.bridge = CvBridge()
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        rospy.loginfo("YOLO Detector Node started, waiting for images...")

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(cv_img)[0]
            vis_img = cv_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("YOLO Detection", vis_img)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"YOLO detection error: {e}")

if __name__ == '__main__':
    node = YoloDetectorNode()
    rospy.spin()
