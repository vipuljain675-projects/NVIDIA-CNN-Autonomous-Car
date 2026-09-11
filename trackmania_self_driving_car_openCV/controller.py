import keyboard
import decimal
import time


def frange(first, stop, step):
    r = len(str(step))
    first = decimal.Decimal(first)
    stop = decimal.Decimal(stop)
    step = decimal.Decimal(step)
    list = [round(float(first), r)]
    while abs(first) <= abs(stop):
        first = first + step
        list.append(round(float(first), r))
    return list


def drive(P, N):
    j = 0.001
    slp = 0.045
    slp1 = 0.09
    slp2 = 0.01
    slp3 = 0.2
    if (round(abs(P/N), 3) in frange(0.201, 1.5, j)) or (round(abs(N/P), 3) in frange(0.6, 1.5, j)):
        keyboard.press('w')
        time.sleep(0.03)
        keyboard.release('w')
        print('Straight = {}, {}'.format(round(abs(P/N), 3), round(abs(N/P), 3)))

    elif round(abs(N/P), 3) in frange(0.0, 0.17, j):
        keyboard.press('a + w')
        time.sleep(slp3)
        keyboard.release('a + w')
        print('Hard left = {}'.format(round(abs(N/P), 3)))

    elif round(abs(N/P), 3) in frange(0.171, 0.599, j):
        keyboard.press('a + w')
        time.sleep(0.008)
        keyboard.release('a + w')
        keyboard.press('w')
        time.sleep(slp)
        keyboard.release('w')
        print('Slight left = {}'.format(round(abs(N / P), 3)))

    elif round(abs(P/N), 3) in frange(0.0, 0.05, j):
        keyboard.press('d + w')
        time.sleep(slp1)
        keyboard.release('d + w')
        print('Hard right = {}'.format(round(abs(P/N), 3)))

    elif round(abs(P/N), 3) in frange(0.051, 0.2, j):
        keyboard.press('d + w')
        time.sleep(0.008)
        keyboard.release('d + w')
        keyboard.press('w')
        time.sleep(slp)
        keyboard.release('w')

        print('Slight right = {}'.format(round(abs(P/N), 3)))

    else:
        print('no action found --> P = {}, N = {}, P/N = {}'.format(P, N, round(abs(P/N), 3)))
