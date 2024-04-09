import cv2 as cv
from utils.arg_parse import get_args
from utils.fps_calc import CvFpsCalc
from utils.ui_tools import draw_info, draw_landmarks
from detector import Detector
from mode_mouse import MouseHandler, play_transition_animation
import autopy


def main():
    frameR = 100
    args = get_args()
    mode = "free"
    angle = 0
    space_pressed = False
    predict_status = True
    exit_mouse = False
    toggle_down = False
    id_prev_pos = (0, 0)

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    wScr, hScr = autopy.screen.size()
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    detector = Detector()
    mouse_handler = MouseHandler(frameR, args.width, args.height, wScr, hScr)

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
        else:
            if mode == "move":
                lmlist, bbox = detector.get_all_position(image)
                id_curr_pos = detector.find_id_position(image, 8, id_prev_pos)
                id_prev_pos = id_curr_pos
                if angle < 360:
                    angle = play_transition_animation(image, id_curr_pos, angle)
                elif angle == 360:
                    if lmlist:
                        fingers = detector.fingers_up(lmlist)
                        exit_mouse, toggle_down = mouse_handler.control(lmlist, fingers, exit_mouse)
                        cv.rectangle(image, (frameR, frameR), (args.width - frameR, args.height - frameR),
                                     (255, 255, 255), 2)
                        if id_curr_pos:
                            if not toggle_down:
                                cv.circle(image, id_curr_pos, 10, (255, 255, 255), cv.FILLED)
                            else:
                                cv.rectangle(image, (id_curr_pos[0] - 10, id_curr_pos[1] - 10),
                                             (id_curr_pos[0] + 10, id_curr_pos[1] + 10), (255, 255, 255), cv.FILLED)
                if exit_mouse:
                    predict_status = True
                    exit_mouse = False
            elif mode == "record":
                landmark_list = detector.get_landmarks(image)
                draw_landmarks(image, landmark_list)
                if space_pressed:
                    detector.record_data(key, 6, 1, landmark_list, image)
                    space_pressed = False

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
