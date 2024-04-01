import cv2 as cv
from utils.arg_parse import get_args
from utils.fps_calc import CvFpsCalc
from utils.ui_tools import draw_info
import autopy
from detector import Detector


class MouseHandler:
    def __init__(self):
        self.smoothening = 7
        self.prev_pos = (0, 0)
        self.curr_pos = (0, 0)

    def calc_movement(self, prev_pos, curr_pos):
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        x_offset = curr_x - prev_x
        y_offset = curr_y - prev_y
        return x_offset, y_offset

    def map_hand_display(self, offset_x, offset_y):
        mouse_x, mouse_y = autopy.mouse.location()
        move_x = mouse_x + offset_x
        move_y = mouse_y + offset_y
        return move_x, move_y

    def mouse_control(self, curr_pos):
        x_offset, y_offset = self.calc_movement(self.prev_pos, curr_pos)
        move_x, move_y = self.map_hand_display(x_offset, y_offset)
        curr_x, curr_y = curr_pos
        clocX = curr_x + (move_x - curr_x) / self.smoothening
        clocY = curr_y + (move_y - curr_y) / self.smoothening
        autopy.mouse.move(clocX, clocY)
        self.prev_pos = curr_pos


def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    detector = Detector()
    mouse_handler = MouseHandler()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
            break

        image = cv.flip(image, 1)
        image, gesture_name = detector.detect(image)
        mode = map_gesture_mode(gesture_name)

        if mode == "move":
            curr_pos = detector.find_position(image, 8)
            mouse_handler.mouse_control(curr_pos)

        image = draw_info(image, fps)
        cv.imshow("Hand Detection", image)

    cap.release()
    cv.destroyAllWindows()


def map_gesture_mode(gesture_name):
    if gesture_name == "Open":
        mode = "move"
    else:
        mode = "free"
    return mode


if __name__ == "__main__":
    main()
