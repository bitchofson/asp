import cv2 as cv
import numpy as np
import asyncio

class MultiCameraCapture:

    def __init__(self, sources: dict) -> None:
        assert sources

        self.frame_time = 0
        self.captures = {}
        for camera_name, link in sources.items():
            capture = cv.VideoCapture(link)
            assert capture.isOpened()
            self.captures[camera_name] = capture
            cv.namedWindow(camera_name)
    
    @staticmethod
    async def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print('Empty frame')
            return
        return frame
    
    @staticmethod
    async def show_frame(window_name: str, frame: np.array):
        cv.imshow(window_name, frame)
        asyncio.sleep(0.001)
    
    async def async_camera_gen(self):
        for camera_name, capture in self.captures.items():
            yield camera_name, capture, self.frame_time
            self.frame_time += 1 / capture.get(cv.CAP_PROP_FPS)
            await asyncio.sleep(0.001)