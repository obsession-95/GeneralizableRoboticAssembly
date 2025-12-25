import pyrealsense2 as rs
import numpy as np
from pyzed import sl
import cv2

class CameraReader:
    def __init__(self, isRealsense=False, isZed=False):
        self.isRealsense = isRealsense
        self.isZed = isZed
        if isRealsense:
            # RealSense初始化
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.rs_pipeline.start(self.rs_config)

        if isZed:
            # ZED初始化
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise Exception("无法打开ZED相机")
            
    def get_realsense(self):
        # RealSense读取帧
        rs_frames = self.rs_pipeline.wait_for_frames()
        rs_color_frame = rs_frames.get_color_frame()
        rs_image = np.asanyarray(rs_color_frame.get_data())

        return rs_image
    
    def get_zed(self):
        # ZED读取帧
        zed_image = sl.Mat()
        if self.zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed_rgb_image = zed_image.get_data()  # 这里得到的是numpy array格式
            # 确保图像格式是BGR
            if len(zed_rgb_image.shape) > 2 and zed_rgb_image.shape[2] == 4:  # 如果是BGRA
                zed_rgb_image = cv2.cvtColor(zed_rgb_image, cv2.COLOR_BGRA2BGR)
        else:
            zed_rgb_image = None  # 或者你可以选择抛出异常，取决于你的需求
        
        return zed_rgb_image
    
    def get_images(self):
        rs_img, zed_img = None, None
        if self.isRealsense:
            rs_img = self.get_realsense()
        if self.isZed:
            zed_img = self.get_zed()

        return rs_img, zed_img


    def close(self):
        # 清理
        if self.isRealsense:
            self.rs_pipeline.stop()
        if self.isZed:
            self.zed.close()

def main():
    camera = CameraReader(isRealsense=True, isZed=True)
    for _ in range(10):  # 循环10次
        try:
            # 获取两台相机的RGB图像
            rs_image, zed_rgb_image = camera.get_images()
            # 显示图像（可选）
               
            if zed_rgb_image is not None:
                cv2.imshow('RealSense RGB Stream', rs_image)
                cv2.imshow('ZED RGB Stream', zed_rgb_image)
                cv2.waitKey(1000)
            else:
                print("未能成功获取ZED相机的帧")
              # 每一帧显示后暂停1秒再执行
        finally:
            cv2.destroyAllWindows() # 每次显示完图像后都关闭窗口
    camera.close()

if __name__ == "__main__":
    main()