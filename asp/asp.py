import cv2 as cv
import asyncio
from async_frame_reader.video_async import MultiCameraCapture
from utils.setting_camera import add_timestamp_to_frame
from detection.motion_detection import MotionDetector

class App:
    def __init__(self, captured_obj: MultiCameraCapture, camera_processors: dict):
        self.captured_obj = captured_obj
        self.camera_processors = camera_processors

    async def run_task(self, frame, frame_time, camera_processor: MotionDetector, camera_name: str):
        task1 = asyncio.create_task(add_timestamp_to_frame(frame, camera_name))
        task2 = asyncio.create_task(camera_processor.run_detection(frame, frame_time))
        await asyncio.gather(task1, task2)
        await asyncio.sleep(0.01)

    async def main_loop(self):
        while True:
            async for camera_name, cap, frame_time in self.captured_obj.async_camera_gen():
                frame = await self.captured_obj.read_frame(cap)
                camera_processor = self.camera_processors[camera_name]

                await self.run_task(frame, frame_time, camera_processor, camera_name)
                await self.captured_obj.show_frame(camera_name, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()