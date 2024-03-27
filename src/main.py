import cv2 as cv
from utils.arg_parse import get_args
from utils.fps_calc import CvFpsCalc
from utils.ui_tools import select_mode, draw_info
from detector import Detector


def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    detector = Detector()

    # set init mode and number
    mode, number = 0, -1

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode, number)

        image = cv.flip(image, 1)
        image, landmark_list = detector.detect(image, number, mode)
        if key == 32:  # Space key to record data
            detector.record_data(key, number, mode, landmark_list, image)

        image = draw_info(image, fps, mode, number)
        cv.imshow("Hand Detection", image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
