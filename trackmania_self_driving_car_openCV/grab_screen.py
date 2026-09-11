import cv2
import numpy as np

try:
    import win32gui, win32ui, win32con, win32api  # type: ignore # noqa: F401
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False
    try:
        from PIL import ImageGrab
    except ImportError:
        ImageGrab = None


def grab(region=None):
    if not IS_WINDOWS:
        if ImageGrab is not None:
            if region:
                img = ImageGrab.grab(bbox=(region[0], region[1], region[2], region[3]))
            else:
                img = ImageGrab.grab()
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return np.zeros((360, 640, 3), dtype=np.uint8)

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
