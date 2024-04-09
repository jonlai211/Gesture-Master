import cv2 as cv
import numpy as np
import autopy
import time
import math
import logging


class MouseHandler:
    def __init__(self, frameR, wCam, hCam, wScr, hScr):
        self.smoothening = 7
        self.frameR = frameR
        self.hScr = hScr
        self.wCam = wCam
        self.hCam = hCam
        self.wScr = wScr
        self.prev_pos = (0, 0)
        self.last_click_time = 0
        self.click_interval = 1
        self.toggle_down = False

    def control(self, lmList, fingers, exit_mouse):
        if len(lmList) != 0:
            x0, y0 = lmList[4][1:]
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            if fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 0:
                x_screen = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                y_screen = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
                clocX = self.prev_pos[0] + (x_screen - self.prev_pos[0]) / self.smoothening
                clocY = self.prev_pos[1] + (y_screen - self.prev_pos[1]) / self.smoothening
                # logging.info(f"toggle down {str(self.toggle_down)}")
                autopy.mouse.toggle(down=self.toggle_down)
                autopy.mouse.move(clocX, clocY)
                self.prev_pos = clocX, clocY
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 0:
                distance_victory = math.hypot(x2 - x1, y2 - y1)
                if distance_victory < 40:
                    current_time = time.time()
                    if current_time - self.last_click_time >= self.click_interval:
                        autopy.mouse.click()
                        self.last_click_time = current_time
            elif fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 1:
                current_time = time.time()
                if current_time - self.last_click_time >= self.click_interval:
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                    self.last_click_time = current_time
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 1:
                current_time = time.time()
                if current_time - self.last_click_time >= self.click_interval:
                    self.toggle_down = not self.toggle_down
                    self.last_click_time = current_time
            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 1:
                exit_mouse = True
        return exit_mouse, self.toggle_down


def play_transition_animation(image, curr_pos, angle):
    radius = 15
    color = (255, 255, 255)
    thickness = 3
    cv.ellipse(image, (curr_pos[0], curr_pos[1]), (radius, radius), 0, 0, angle, color, thickness)
    angle += 10
    if angle > 360:
        angle = 360
    return angle
