import json
import typing

def load_cameras(path_to_json_file: str) -> typing.Any:
    cameras = json.loads(open(path_to_json_file, 'r').read())
    return cameras
