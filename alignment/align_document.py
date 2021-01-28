import argparse
import cv2

from alignment.align_images import align_images

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
template = cv2.imread(args['template'])

aligned = align_images(image, template, debug=True)

cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("original image", 480, 640)
cv2.imshow("original image", image)
cv2.waitKey(0)

cv2.namedWindow("aligned", cv2.WINDOW_NORMAL)
cv2.resizeWindow("aligned", 480, 640)
cv2.imshow("aligned", aligned)
cv2.waitKey(0)
