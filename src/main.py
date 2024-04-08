import cv2 as cv
import numpy as np
import time
from utils.arg_parse import get_args
from utils.fps_calc import CvFpsCalc
from utils.ui_tools import draw_info, draw_landmarks
import autopy
from detector import Detector
from mode_mouse import MouseHandler, play_transition_animation


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
    predict_status = True
    angle = 0
    id_prev_pos = 0, 0

    mode = "free"

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
        if predict_status:
            image, gesture_name = detector.detect(image)
            mode, angle, predict_status = mode_manager(gesture_name, angle, predict_status)
            # predict_status = False
        else:
            if mode == "move":
                lmlist, bbox = detector.get_all_position(image)
                id_curr_pos = detector.find_id_position(image, 8, id_prev_pos)
                id_prev_pos = id_curr_pos
                if angle < 360:
                    angle = play_transition_animation(image, id_curr_pos, angle)
                elif angle == 360:
                    if len(lmlist) != 0:
                        fingers = detector.fingers_up(lmlist)
                        mouse_handler.move_and_click(lmlist, fingers)
                        if id_curr_pos:
                            cv.circle(image, id_curr_pos, 5, (0, 0, 255), cv.FILLED)

                    # mouse_handler.prev_pos = mouse_handler.mouse_move(id_curr_pos)
                    cv.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # mode = "record"

        # if mode == "move":
        #     id_curr_pos = detector.find_position(image, 8, id_prev_pos)
        #     id_prev_pos = id_curr_pos
        #     if angle < 360:
        #         angle = play_transition_animation(image, id_curr_pos, angle)
        #     elif angle == 360:
        #         mouse_handler.prev_pos = mouse_handler.mouse_move(id_curr_pos)
        #         cv.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # elif mode == "click":
        #     current_time = time.time()
        #     if current_time - mouse_handler.last_click_time >= mouse_handler.click_interval:
        #         autopy.mouse.click()
        #         mouse_handler.last_click_time = current_time
        # elif mode == "record":
        #     landmark_list = detector.get_landmarks(image)
        #     draw_landmarks(image, landmark_list)
        #     if space_pressed:
        #         detector.record_data(key, 6, 1, landmark_list, image)
        #         space_pressed = False

        image = draw_info(image, fps)
        cv.imshow("Gesture Master", image)

    cap.release()
    cv.destroyAllWindows()


def mode_manager(gesture_name, angle, predict_status):
    if gesture_name == "Point":
        mode = "move"
        predict_status = False
    else:
        mode = "free"
        angle = 0
        predict_status = True
    return mode, angle, predict_status


if __name__ == "__main__":
    main()
