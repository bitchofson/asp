import datetime as dt
import cv2 as cv

def add_timestamp_to_frame(frame):
    font = cv.FONT_HERSHEY_PLAIN
    now = dt.datetime.now()
    formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")

    frame = cv.putText(frame, formatted_now, (0, 50), font, 2, (0, 0, 255), 0, cv.LINE_8)
    return frame