from typing import Optional
import cv2
from PIL import Image

class VideoReader:
    def __init__(self):
        super().__init__()

        self.capture = cv2.VideoCapture()

    def open(self, video_file : str):
        self.capture.open(video_file)
        self.frame_num = 0

    def get_frame(self) -> Optional[Image.Image]:
        ret_val, frame = self.capture.read()

        if not ret_val:
            return None

        self.frame_num += 1
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def seek(self, frame_num : int):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.frame_num = frame_num

    def close(self):
        self.capture.release()
