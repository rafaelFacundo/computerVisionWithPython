import cv2
import numpy as np
import machine

def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and \
        (y1+h1 < y2+h2)

def wrap_digit(rect, img_w, img_h):
    x, y, w, h = rect
    x_center = x + w//2
    y_center = y + h//2
    if (h > w):
        w = h
        x = x_center - (w//2)
    else:
        h = w
        y = y_center - (h//2)
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    if x < 0:
        x = 0
    elif x > img_w:
        x = img_w
    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h
    if x+w > img_w:
        w = img_w - x
    if y+h > img_h:
        h = img_h - y
    return x, y, w, h

ann, test_data = machine.train(
    machine.create_ann(60), 50000, 10)