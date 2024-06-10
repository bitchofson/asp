import asyncio
import asp
from async_frame_reader.video_async import MultiCameraCapture
from utils.cameras_load import load_cameras
from detection.background_subtraction import BackgroundSubtraction
from detection.optical_flow import OpticalFlow


if __name__ == '__main__':
    path_to_json_setting = 'resources/settings.json'

    settings = load_cameras(path_to_json_setting)
    sources = settings['cameras']
    records = settings['records']
    path_to_save = settings['path_to_save_recording']
    tracking_areas = settings['tracking_areas']
    detector_types = settings['detector_types']

    captured = MultiCameraCapture(sources=sources)

    camera_processors = {}
    for record_name, camera_name in records.items():
        pts = tracking_areas[camera_name]
        detector_type = detector_types[camera_name]
        if detector_type == 'opticalflow':
            camera_processors[camera_name] = OpticalFlow(pts=pts, path_to_save=path_to_save)
        elif detector_type == 'backgroundsubtraction':
            camera_processors[camera_name] = BackgroundSubtraction(pts=pts, path_to_save=path_to_save)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    app = asp.App(captured, camera_processors)
    asyncio.run(app.main_loop())