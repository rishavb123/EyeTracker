import cv2
import numpy

from cv2utils.args import make_parser
from cv2utils.camera import make_camera_with_args

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

parser = make_parser()
camera, args = make_camera_with_args(parser=parser, log=True, fps=30, res=(1920, 1080), src=1)

camera.stream()