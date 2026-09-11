import time
from statistics import StatisticsError, mean
import cv2
import numpy as np

try:
    from grab_screen import grab  # type: ignore
    from controller import drive  # type: ignore
except ImportError:
    from trackmania_self_driving_car_openCV.grab_screen import grab  # type: ignore
    from trackmania_self_driving_car_openCV.controller import drive  # type: ignore


def process_frame(frame):
    frame = cv2.resize(frame, (640, 360))
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = np.array([[0, 150], [640, 150], [640, 360], [580, 360], [360, 220], [280, 220], [60, 360], [0, 360], [0, 310]])
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    frame = cv2.Canny(frame, 270, 350)
    frame = roi(frame, [pts])
    lines = cv2.HoughLinesP(frame, 1, np.pi/180, 50, np.array([]), 10, 20)
    slopes = []
    if lines is not None:
        for line in lines:
            xy = line[0]
            slopes.append((xy[3]-xy[1])/(xy[2]-xy[0]))
            cv2.line(frame2, (xy[0], xy[1]), (xy[2], xy[3]), [255, 0, 0], 4)

    pslope = [i for i in slopes if i >= 0]
    nslope = [i for i in slopes if i < 0]
    try:
        mean_pslope = round(mean(pslope), 3) if pslope else 0.0
    except (StatisticsError, ValueError):
        mean_pslope = 0.0

    try:
        mean_nslope = round(mean(nslope), 3) if nslope else 0.0
    except (StatisticsError, ValueError):
        mean_nslope = 0.0
    return frame2, mean_pslope, mean_nslope


def roi(frame, pts):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, pts, 255)
    masked = cv2.bitwise_and(frame, mask)
    return masked


for i in range(4, 0, -1):
    print(i)
    time.sleep(1)


def main():
    paused = False
    # esc key to exit
    while True:
        if not paused:

            frame = grab((0, 45, 780, 485))
            frm = process_frame(frame)
            frame = frm[0]
            drive(frm[1], frm[2])
            cv2.imshow('window', frame)
            #cv2.imshow('edges', frame[1])
            key = cv2.waitKey(25) & 0xFF
            if key == 27:
                break
            elif key == 32:
                paused = not paused
        else:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == 27:
                break
            elif key2 == 32:
                paused = not paused
    cv2.destroyAllWindows()


while True:
    try:
        main()
    except TypeError:
        drive(10, 10)
        continue
    cv2.destroyAllWindows()
