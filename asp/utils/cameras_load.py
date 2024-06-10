import json
import typing

def load_cameras(path_to_json_file: str) -> typing.Any:
    with open(path_to_json_file, 'r') as f:
        settings = json.load(f)['setting']

    cameras = settings[0]['cameras']
    records = settings[1]['records']
    path_to_save = settings[2]['path_to_save_recording']
    tracking_areas = settings[3]['tracking_areas']
    detector_types = settings[4]['detector_types']

    return {
        'cameras': cameras,
        'records': records,
        'path_to_save_recording': path_to_save,
        'tracking_areas': tracking_areas,
        'detector_types': detector_types
    }
