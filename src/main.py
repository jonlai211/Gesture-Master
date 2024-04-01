import cv2 as cv
import numpy as np
import time
from utils.arg_parse import get_args
from utils.fps_calc import CvFpsCalc
from utils.ui_tools import draw_info, draw_landmarks
import autopy
from detector import Detector


class MouseHandler:
    def __init__(self, frameR, wCam, hCam, wScr, hScr):
        self.smoothening = 7
        self.frameR = frameR
        self.wCam = wCam
        self.hCam = hCam
        self.wScr = wScr
        self.hScr = hScr
        self.prev_pos = (0, 0)
        self.curr_pos = (0, 0)
        self.last_click_time = 0
        self.click_interval = 1

    def mouse_move(self, curr_pos):
        x1 = curr_pos[0]
        y1 = curr_pos[1]
        # print('x1: ', x1, ';y1: ', y1)
        x2 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
        y2 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
        # print(self.wCam - self.frameR)
        # print('x2: ', x2, ';y2: ', y2)
        clocX = self.prev_pos[0] + (x2 - self.prev_pos[0]) / self.smoothening
        clocY = self.prev_pos[1] + (y2 - self.prev_pos[1]) / self.smoothening
        # print('clocX: ', clocX, ';clocY: ', clocY)
        autopy.mouse.move(clocX, clocY)
        self.prev_pos = clocX, clocY
        return self.prev_pos

    def mouse_click(self):
        autopy.mouse.click()


def main():
    frameR = 100
    wCam, hCam = 640, 480
    wScr, hScr = autopy.screen.size()

    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    detector = Detector()
    mouse_handler = MouseHandler(frameR, wCam, hCam, wScr, hScr)
    space_pressed = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
            break
        elif key == 32:
            space_pressed = not space_pressed

        image = cv.flip(image, 1)
        image, gesture_name = detector.detect(image)
        mode = map_gesture_mode(gesture_name)
        # mode = "move"

        if mode == "move":
            curr_pos = detector.find_position(image, 8)
            mouse_handler.prev_pos = mouse_handler.mouse_move(curr_pos)
            cv.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        elif mode == "click":
            current_time = time.time()
            if current_time - mouse_handler.last_click_time >= mouse_handler.click_interval:
                autopy.mouse.click()
                mouse_handler.last_click_time = current_time
        elif mode == "record":
            landmark_list = detector.get_landmarks(image)
            draw_landmarks(image, landmark_list)
            if space_pressed:
                detector.record_data(key, 15, 1, landmark_list, image)
                space_pressed = False

        image = draw_info(image, fps)
        cv.imshow("Hand Detection", image)

    cap.release()
    cv.destroyAllWindows()


def map_gesture_mode(gesture_name):
    if gesture_name == "Point":
        mode = "move"
    elif gesture_name == "Rock":
        mode = "click"
    else:
        mode = "free"
    return mode


if __name__ == "__main__":
    main()
