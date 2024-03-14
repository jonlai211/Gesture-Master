import cv2
import mediapipe as mp
from utils.ui_tools import calc_bounding_rect, calc_landmark_list, draw_landmarks, draw_bounding_rect, draw_info_text
from utils.csv_tools import logging_csv


class Detector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.results = None
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, image, number, mode):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)

        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks,
                                                  self.results.multi_handedness):
                # calculate bounding rect
                bounding_rect = calc_bounding_rect(image, hand_landmarks)
                # calculate landmark
                landmark_list = calc_landmark_list(image, hand_landmarks)

                # save landmark to csv
                logging_csv(number, mode, landmark_list)

                # draw landmark
                image = draw_bounding_rect(bounding_rect, image, bounding_rect)
                image = draw_landmarks(image, landmark_list)
                image = draw_info_text(
                    image,
                    bounding_rect,
                    handedness,
                    "Test"
                )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image
