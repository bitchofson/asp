import cv2 as cv

from async_frame_reader.video_async import MultiCameraCapture
from utils.setting_camera import add_timestamp_to_frame
from utils.cameras_load import load_cameras
from detection.background_subtraction import run_detection_mog

def main():
    path_to_json_setting = 'resources/settings.json'

    settings = load_cameras(path_to_json_setting)['setting']

    captured = MultiCameraCapture(sources=settings[0])
    
    
    while True:
        for camera_name, cap in captured.captures.items():
            frame = captured.read_frame(cap)
            frame = add_timestamp_to_frame(frame=frame)

            if camera_name in settings[1].values():
                frame = run_detection_mog(frame=frame, pts=captured.pts, path_to_save=settings[2]['path_to_save_recording'])

            cv.imshow(f'{camera_name}', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
