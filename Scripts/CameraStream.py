import cv2

DEVICE = '/dev/video0'
SIZE = (640, 480)
FILENAME = 'capture.png'


def camstream():
    cap = cv2.VideoCapture(0)
    condition = True
    while condition:
        condition, img = cap.read()
        cv2.imshow('image', img)
    return img


if __name__ == '__main__':
    camstream()

