import cv2 as cv
import numpy as np
import datetime as dt

out_frame = None
current_recording_name = None
detect = False
non_detected_counter = 0
backSub = cv.createBackgroundSubtractorMOG2()

# Функция определяющая нахождение точки внутри области отслеживания
def inside_polygon(point, polygon) -> bool:
    result = cv.pointPolygonTest(polygon, (point[0], point[1]), False)
    if result == 1:
        return True
    else:
        return False

def run_detection_mog(frame, pts, path_to_save):

    global non_detected_counter
    global detect
    global current_recording_name
    global out_frame

    fg_mask = backSub.apply(frame)

    retval, mask_thresh = cv.threshold(fg_mask, 180, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    mask_eroded = cv.morphologyEx(mask_thresh, cv.MORPH_OPEN, kernel)

    contours, hierarchy = cv.findContours(mask_eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    min_contour_area = 500  

    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    
    frame_copy = frame.copy()
    for cnt in large_contours:
        center_x = None
        center_y = None
        
        x, y, w, h = cv.boundingRect(cnt)

        center_x = int((x + (x + w)) / 2)
        center_y = int((y + (y + h)) / 2)

        cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv.circle(frame_copy, (center_x, center_y), 5, (0, 0, 255), -1)

        
        if len(pts) >= 4:
            cv.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
            cv.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
            if center_x is not None and center_y is not None:
                if inside_polygon((center_x, center_y), np.array([pts])):
                    non_detected_counter = 0
                    detect = True
                    if out_frame is None:  
                            now = dt.datetime.now()
                            formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                            print("Motion detected at", formatted_now)
                            current_recording_name = f'{path_to_save}{formatted_now}.mp4'
                            fourcc = cv.VideoWriter_fourcc(*'mp4v')
                            out_frame = cv.VideoWriter(current_recording_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                            
    non_detected_counter += 1
    if non_detected_counter >= 200:
        print(f'{non_detected_counter}')
        non_detected_counter = 0
        detect = False


    if detect:
        print('Write video...')
        out_frame.write(frame)
    else:
        if out_frame is not None:
            print('Record release...')  
            out_frame.release()
            out_frame = None  
            current_recording_name = None

    return frame_copy