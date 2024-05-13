import threading
import time
import datetime
import numpy as np
import cv2



class Camera:

    out_frame = None
    cap = cv2.VideoCapture(2)

    def __init__(self, path_output: str, window_name: str) -> None:

        # Объект потока для работы с камерой
        self.__camera_thread = None
        
        # Режим охраны
        self.armed = False

        # Название окна
        self.__window_name = window_name
        
        # Путь для сохранения файлов 
        self.__path_output = path_output

        # Точки для задания области отслеживания
        self.pts = [[0, 0], [600, 0], [600, 200], [0, 200]]

        self.backSub = cv2.createBackgroundSubtractorMOG2()
    
    # Функция задающая область отслеживания
    def __draw_polygon(self,event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pts = []

    # Функция определяющая нахождение точки внутри области отслеживания
    def inside_polygon(self, point, polygon) -> bool:
        result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
        if result == 1:
            return True
        else:
            return False
    
    # Функция включения камеры
    def arm(self) -> None:
        if not self.armed and not self.__camera_thread:
            self.__camera_thread = threading.Thread(target=self.run)
        self.__camera_thread.start()
        self.armed = True
        print("Camera armed...")
    
    # функция выключения камеры
    def disarm(self) -> None:
        self.armed = False
        self.__camera_thread = None
        print("Camera disarmed....")

    # Основной поток событий
    def run(self):

        non_detected_counter = 0
        current_recording_name = None
        detect = False

        #cv2.namedWindow(self.__window_name)
        #cv2.setMouseCallback(self.__window_name, self.__draw_polygon)
        
        #Camera.cap = cv2.VideoCapture(2)

        print('Camera started...')
        while self.armed:
            _, raw_frame = self.cap.read()
            fg_mask = self.backSub.apply(raw_frame)

            retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 500  

            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            frame_copy = raw_frame.copy()
            for cnt in large_contours:
                center_x = None
                center_y = None
                
                x, y, w, h = cv2.boundingRect(cnt)

                center_x = int((x + (x + w)) / 2)
                center_y = int((y + (y + h)) / 2)

                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

                cv2.circle(frame_copy, (center_x, center_y), 5, (0, 0, 255), -1)

                
                if len(self.pts) >= 4:
                    cv2.fillPoly(frame_copy, np.array([self.pts]), (0, 255, 0))
                    cv2.addWeighted(frame_copy, 0.1, raw_frame, 0.9, 0)
                    if center_x is not None and center_y is not None:
                        if self.inside_polygon((center_x, center_y), np.array([self.pts])):
                            non_detected_counter = 0
                            detect = True
                            if self.out_frame is None:  
                                    now = datetime.datetime.now()
                                    formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                                    print("Motion detected at", formatted_now)
                                    current_recording_name = f'{self.__path_output}{formatted_now}.mp4'
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    self.out_frame = cv2.VideoWriter(current_recording_name, fourcc, 30.0, (raw_frame.shape[1], raw_frame.shape[0]))
                                    
            non_detected_counter += 1
            if non_detected_counter >= 200:
                non_detected_counter = 0
                detect = False


            if detect:
                self.out_frame.write(raw_frame)
            else:
                if self.out_frame is not None:  
                    print('Record release...')
                    self.out_frame.release()
                    self.out_frame = None  
                    current_recording_name = None
            
            #cv2.imshow(self.__window_name, frame_copy)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        if self.out_frame is not None:
            self.out_frame.release()
            self.out_frame = None
            current_recording_name = None

        self.cap.release()
        print('Camera released...')
        #cv2.destroyAllWindows()


    def __del__(self):
        self.cap.release()
        if self.out_frame is not None:
            self.out_frame.release()