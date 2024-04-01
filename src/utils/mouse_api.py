import autopy
import time


class MouseAPI:
    def __init__(self):
        self.FAILSAFE = True
        self.autopy = autopy

    def move(self, x_offset, y_offset):
        self.autopy.mouse.move(x_offset, y_offset)

    def moveTo(self, x, y):
        screen_width, screen_height = self.getScreenSize()
        if 0 <= x < screen_width and 0 <= y < screen_height:
            self.autopy.mouse.move(x, y)
        else:
            print("Error: Target coordinates out of bounds.")

    def click(self, button='left'):
        if button == 'left':
            self.autopy.mouse.click(self.autopy.mouse.Button.LEFT)
        elif button == 'right':
            self.autopy.mouse.click(self.autopy.mouse.Button.RIGHT)

    def dragTo(self, x, y, duration=1, button='left'):
        if button == 'left':
            self.autopy.mouse.toggle(self.autopy.mouse.Button.LEFT, True)
            self.moveTo(x, y)
            self.autopy.mouse.toggle(self.autopy.mouse.Button.LEFT, False)
        elif button == 'right':
            self.autopy.mouse.toggle(self.autopy.mouse.Button.RIGHT, True)
            self.moveTo(x, y)
            self.autopy.mouse.toggle(self.autopy.mouse.Button.RIGHT, False)

    def drag(self, x_offset, y_offset, duration=1, button='left'):
        current_x, current_y = self.getPosition()
        new_x, new_y = current_x + x_offset, current_y + y_offset
        self.dragTo(new_x, new_y, duration, button=button)

    def scroll(self, amount):
        self.autopy.mouse.scroll(amount)

    def getPosition(self):
        return self.autopy.mouse.location()

    def getScreenSize(self):
        return self.autopy.screen.size()

    def smoothMoveTo(self, x, y, steps=50, delay=0.02):
        current_x, current_y = self.getPosition()
        dx = (x - current_x) / steps
        dy = (y - current_y) / steps
        for _ in range(steps):
            current_x += dx
            current_y += dy
            self.move(current_x, current_y)
            time.sleep(delay)


if __name__ == "__main__":
    mouse = MouseAPI()
    target_x, target_y = 1466.8, 45.6
    offset_x, offset_y = 100, 100
    print("Before move:", mouse.getPosition())

    mouse.smoothMoveTo(target_x, target_y)

    mouse.click(button='left')
    print("After move:", mouse.getPosition())
