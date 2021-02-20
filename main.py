import cv2
import numpy as np

from cv2utils.args import make_parser
from cv2utils.camera import make_camera_with_args

face_cascade = cv2.CascadeClassifier(
    "./classifiers/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("./classifiers/haarcascade_eye.xml")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 2000
detector = cv2.SimpleBlobDetector_create(detector_params)


def preprocess(frames, raw):
    frame = frames[0]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = list(face_cascade.detectMultiScale(gray))

    sorted(faces, key=lambda face: -face[2] * face[3])

    for x, y, w, h in faces[:num_of_people]:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = frame[y : y + h, x : x + w]
        gray_face = gray[y : y + h * 2 // 3, x : x + w]
        eyes = list(eye_cascade.detectMultiScale(gray_face))
        eyes = filter(lambda eye: eye[2] < w * 3 / 8, eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 2)
            eye = face[ey : ey + eh, ex : ex + ew]
            img = gray_face[ey : ey + eh, ex : ex + ew]
            # img = cv2.GaussianBlur(gray_eye, (15, 15), 0)
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # img = cv2.filter2D(img, -1, kernel)
            _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            img = cv2.erode(img, None, iterations=2)  # 1
            img = cv2.dilate(img, None, iterations=4)  # 2
            img = cv2.medianBlur(img, 5)  # 3
            # face[ey : ey + eh, ex : ex + ew] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            keypoints = detector.detect(img)
            print(keypoints)
            cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255))

    frame = cv2.resize(frame, dsize=(1920, 1080))

    return frame


parser = make_parser()
parser.add_argument(
    "-p",
    "--people",
    type=int,
    default=1,
    help="The number of people in the video stream",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=int,
    default=0,
    help="The threshold to run on the image before applying blob detection. If it is left as 0 or default, a slider will be used",
)
camera, args = make_camera_with_args(
    parser=parser, log=True, fps=30, res=(640, 360), cam=1
)
camera.name = "Image"
cv2.namedWindow("Image")
num_of_people = args.people
threshold = args.threshold
if threshold == 0:

    def set_threshold(t):
        global threshold
        threshold = t

    cv2.createTrackbar("threshold", "Image", 0, 255, set_threshold)

# def _default_output_function(self, frame):
#     cv2.imshow(self.name, frame)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord("q") or k == 27:
#         return False
#     return True

camera.stream(preprocess=preprocess, frames_stored=1)