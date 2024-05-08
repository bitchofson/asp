import threading
import cv2
import datetime


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
        self.pts = []

        self.person_detected = False
    
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
        self.__camera_thread = threading.Thread(target=self.run)
        self.__camera_thread.start()
        self.armed = True
    
    # функция выключения камеры
    def disarm(self) -> None:
        self.__camera_thread = None
        self.armed = False

    # Основной поток событий
    def run(self):

        non_detected_counter = 0
        current_recording_name = None

        cv2.namedWindow(self.__window_name)
        cv2.setMouseCallback(self.__window_name, self.__draw_polygon)

        print('Camera started...')
        while self.armed:
            _, raw_frame = self.cap.read()

            if self.person_detected:
                non_detected_counter = 0
                if self.out_frame is None:  
                        now = datetime.datetime.now()
                        formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                        print("Person motion detected at", formatted_now)
                        current_recording_name = f'{self.__path_output}{formatted_now}.mp4'
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.out_frame = cv2.VideoWriter(current_recording_name, fourcc, 20.0, (raw_frame.shape[1], raw_frame.shape[0]))
                self.out_frame.write(raw_frame)
            else:
                non_detected_counter += 1
                if non_detected_counter >= 50:  
                    if self.out_frame is not None:  
                        self.out_frame.release()
                        self.out_frame = None  
                        current_recording_name = None

            cv2.imshow(self.__window_name, raw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.out_frame is not None:
            self.out_frame.release()
            self.out_frame = None
            current_recording_name = None

        self.cap.release()
        print('Camera released...')
        cv2.destroyAllWindows()


    def __del__(self):
        self.cap.release()
        if self.out_frame is not None:
            self.out_frame.release()
