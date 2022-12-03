import math

import cv2
import numpy as np


class LineDetector():
    def __init__(self) -> None:
        pass
    

    def PreProcess(func):
        def wrapper(self, *args, **kwargs):
            img = args[0].copy()
            return func(self, img)

        return wrapper


    @PreProcess
    def HoughLineTransformProbabilistic(self, image):
        linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 50, 10)
        return linesP


    @PreProcess
    def HoughLineTransform(self, image):
        lines = cv2.HoughLines(image, 0.5, np.pi / 180, 150, None, 0, 0)
        return lines


    def DrawHoughLines(self, image:np.ndarray, lines, is_probabilistic:bool):
        canvas = image.copy()

        if len(canvas.shape) < 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        if is_probabilistic:
            if lines is not None:
                for i in range(0, len(lines)):
                    l = lines[i][0]
                    canvas = cv2.line(canvas, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

        elif not is_probabilistic:
            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    canvas = cv2.line(canvas, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

        return canvas



