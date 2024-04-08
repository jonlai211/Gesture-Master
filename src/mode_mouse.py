import cv2 as cv
import numpy as np
import autopy
import time


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

    def move_and_click(self, lmList, fingers):
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index finger tip coordinates

            if fingers[1] == 1 and fingers[2] == 0:
                x2 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                y2 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
                clocX = self.prev_pos[0] + (x2 - self.prev_pos[0]) / self.smoothening
                clocY = self.prev_pos[1] + (y2 - self.prev_pos[1]) / self.smoothening
                autopy.mouse.move(clocX, clocY)
                self.prev_pos = clocX, clocY
            elif fingers[1] == 1 and fingers[2] == 1:
                current_time = time.time()
                if current_time - self.last_click_time >= self.click_interval:
                    autopy.mouse.click()
                    self.last_click_time = current_time


def play_transition_animation(image, curr_pos, angle):
    radius = 15
    color = (255, 255, 255)
    thickness = 3
    cv.ellipse(image, (curr_pos[0], curr_pos[1]), (radius, radius), 0, 0, angle, color, thickness)
    angle += 10
    if angle > 360:
        angle = 360
    return angle
