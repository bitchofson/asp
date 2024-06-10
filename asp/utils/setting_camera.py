import datetime as dt
import cv2 as cv

async def add_timestamp_to_frame(frame, camera_name):
    font = cv.FONT_HERSHEY_PLAIN
    now = dt.datetime.now()
    formatted_now = now.strftime("%d-%m-%y %H:%M:%S")
    
    # Добавляем информацию на кадр
    text = f"{camera_name} | {formatted_now}"
    # Наносим текст на кадр
    frame = cv.putText(frame, text, (10, 20), font, 1, (0, 255, 0), 1, cv.LINE_8)
