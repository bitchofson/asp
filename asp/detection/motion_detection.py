import numpy as np
import cv2 as cv

from abc import ABC, abstractmethod

class MotionDetector(ABC):
    @abstractmethod
    async def run_detection(self, frame, frame_time):
        pass
    
    def inside_polygon(self, point, polygon) -> bool:
        result = cv.pointPolygonTest(polygon, (point[0], point[1]), False)
        return result == 1

    def get_contour_detections(self, mask, thresh=400):
        contours, _ = cv.findContours(mask, 
                                    cv.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                    cv.CHAIN_APPROX_TC89_L1)
        detections = []
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            area = w*h
            if area > thresh: 
                detections.append([x,y,x+w,y+h, area])

        return np.array(detections)

    def remove_contained_bboxes(self, boxes):
        check_array = np.array([True, True, False, False])
        keep = list(range(0, len(boxes)))
        for i in keep: # range(0, len(bboxes)):
            for j in range(0, len(boxes)):
                # check if box j is completely contained in box i
                if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                    try:
                        keep.remove(j)
                    except ValueError:
                        continue
        return keep

    def non_max_suppression(self, boxes, scores, threshold=1e-1):
        boxes = boxes[np.argsort(scores)[::-1]]

        order = self.remove_contained_bboxes(boxes)

        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                            max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                        (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
                iou = intersection / union

                if iou > threshold:
                    order.remove(j)
                    
        return boxes[keep]

    def draw_bboxes(self, frame, detections):
        for det in detections:
            x1,y1,x2,y2 = det
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

    def get_color(self, number):
        blue = int(number*30 % 256)
        green = int(number*103 % 256)
        red = int(number*50 % 256)

        return red, blue, green

    def plot_points(self, image, points, radius=3, color=(0,255,0)):
        for x,y in points:
            cv.circle(image, (int(x), int(y)), radius, color, thickness=-1)

        return image
