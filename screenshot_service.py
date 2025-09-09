import pyautogui
import time


def capture_screenshot():
    """
    Take a screenshot and return it along with epoch timestamp
    """
    screenshot = pyautogui.screenshot()
    epoch_time = int(time.time())
    return screenshot, epoch_time


def resize_screenshot(screenshot, target_width=1512):
    """
    Resize image to have specified width while maintaining aspect ratio
    """
    width, height = screenshot.size
    new_width = target_width
    new_height = int(height * (new_width / width))
    resized_img = screenshot.resize((new_width, new_height))
    return resized_img, new_width, new_height
