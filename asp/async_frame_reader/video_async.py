import cv2 as cv

class MultiCameraCapture:

    def __init__(self, sources: dict) -> None:
        assert sources

        self.captures = {}

        self.pts = []

        for camera_name, link in sources.items():
            capture = cv.VideoCapture(link)
            assert capture.isOpened()
            self.captures[camera_name] = capture
            cv.namedWindow(camera_name)
            cv.setMouseCallback(camera_name, self.draw_polygon)

    
    @staticmethod
    def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print('Empty frame')
            return
        return frame
    
    # Функция задающая область отслеживания
    def draw_polygon(self,event, x, y, flags, param) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
            self.pts.append([x, y])
        elif event == cv.EVENT_RBUTTONDOWN:
            self.pts = []