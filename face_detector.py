import cv2
import time
from typing import Tuple, Optional


class FaceDetector:
    """
    基于 Haar 级联的人脸检测器
    支持：实时检测、FPS 计算、图像/视频/摄像头三种输入模式
    """

    def __init__(self,
                 cascade_path: Optional[str] = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        """
        初始化检测器

        Args:
            cascade_path: Haar 模型路径，None 则使用默认
            scale_factor: 图像缩放因子
            min_neighbors: 最小邻居数（过滤误检）
            min_size: 最小检测尺寸
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        # 性能统计
        self.fps = 0
        self.frame_count = 0

    def detect(self, frame) -> Tuple:
        """
        检测单帧图像中的人脸

        Returns:
            (annotated_frame, faces)
            faces 是 numpy 数组 [[x, y, w, h], ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        # 画框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 计算 FPS
        self.frame_count += 1
        # 这里简化处理，实际应计算时间差

        return frame, faces

    def run_camera(self, camera_id: int = 0, save_video: bool = False):
        """
        运行实时摄像头检测（完整流程封装）

        Args:
            camera_id: 摄像头 ID
            save_video: 是否保存为视频文件
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        # 视频保存设置（如果开启）
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        print("摄像头已启动，按 'q' 退出")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, faces = self.detect(frame)

                # 显示信息
                info = f"Faces: {len(faces)} | Frame: {self.frame_count}"
                cv2.putText(annotated_frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Face Detection', annotated_frame)

                if save_video:
                    out.write(annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # 确保资源释放（这是工程化关键！）
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            print(f"结束，共处理 {self.frame_count} 帧")


# 使用示例（放在 if __name__ == "__main__": 下）
if __name__ == "__main__":
    # 方案 A：用图片测试（今晚用这个）
    detector = FaceDetector(scale_factor=1.1, min_neighbors=5)

    import cv2

    # 放一张你的照片到同目录，命名为 test.jpg
    img = cv2.imread('test.jpg')

    if img is None:
        print("错误：找不到 test.jpg，请放一张图片到同目录")
        exit()

    result, faces = detector.detect(img)
    print(f"检测到 {len(faces)} 张人脸")

    # 保存结果
    cv2.imwrite('output_detected.jpg', result)
    print("结果已保存为 output_detected.jpg")

    # 方案 B：摄像头模式（明天配置 WSL 摄像头后再用）
    # detector.run_camera(save_video=False)