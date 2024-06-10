import cv2 as cv
import numpy as np
import datetime as dt

from .motion_detection import MotionDetector

class BackgroundSubtraction(MotionDetector):
    def __init__(self, pts, path_to_save):
        self.backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)
        self.kernel = np.array((9, 9), dtype=np.uint8)
        self.points = pts
        self.path_to_save = path_to_save
        self.out_frame = None
        self.current_recording_name = None
        self.detect = False
        self.non_detected_counter = 0

    def get_motion_mask(self, fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
        _, thresh = cv.threshold(fg_mask,min_thresh,255,cv.THRESH_BINARY)
        motion_mask = cv.medianBlur(thresh, 3)
        
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel, iterations=1)

        return motion_mask

    def get_detections(self, backSub, frame, bbox_thresh=100, nms_thresh=0.1, kernel=np.array((9,9), dtype=np.uint8)):
        fg_mask = backSub.apply(frame)

        motion_mask = self.get_motion_mask(fg_mask, kernel=kernel)

        detections = self.get_contour_detections(motion_mask, bbox_thresh)

        if len(detections) == 0:
            return []
        bboxes = detections[:, :4]
        scores = detections[:, -1]

        return self.non_max_suppression(bboxes, scores, nms_thresh)

    async def run_detection(self, frame_bgr) -> cv.Mat:
        detections = self.get_detections(self.backSub,
                                frame_bgr,  
                                bbox_thresh=100,
                                nms_thresh=1e-2,
                                kernel=self.kernel)
        self.draw_bboxes(frame_bgr, detections)
        
        for cnt in detections:
            center_x = None
            center_y = None
            
            (x, y, w, h) = cnt
            center_x = int((x + (x + w)) / 2)
            center_y = int((y + (y + h)) / 2)

            cv.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

            if len(self.points) >= 4:
                cv.fillPoly(frame_bgr, np.array([self.points]), (0, 255, 0))
                alpha = 0.3 
                frame_bgr = cv.addWeighted(frame_bgr, alpha, frame_bgr, 1 - alpha, 0)
                if center_x is not None and center_y is not None:
                    if self.inside_polygon((center_x, center_y), np.array([self.points])):
                        self.non_detected_counter = 0
                        self.detect = True
                        if self.out_frame is None:  
                                now = dt.datetime.now()
                                formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                                print("Motion detected at", formatted_now)
                                self.current_recording_name = f'{self.path_to_save}{formatted_now}.mp4'
                                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                                self.out_frame = cv.VideoWriter(self.current_recording_name, fourcc, 30.0, (frame_bgr.shape[1], frame_bgr.shape[0]))
                                
        self.non_detected_counter += 1
        if self.non_detected_counter >= 200:
            print(f'{self.non_detected_counter}')
            self.non_detected_counter = 0
            self.detect = False

        if self.detect:
            self.out_frame.write(frame_bgr)
        else:
            if self.out_frame is not None:
                print('Record release...')  
                self.out_frame.release()
                self.out_frame = None  
                self.current_recording_name = None

        
        return frame_bgr
