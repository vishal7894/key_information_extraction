import cv2
import imutils
import numpy as np


def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    (kpsA, descsA) = orb.detectAndCompute(image_gray, None)
    (kpsB, descsB) = orb.detectAndCompute(template_gray, None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if debug:
        matched_vis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv2.imshow("Matched Keypoint", matched_vis)
        cv2.waitKey(0)

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for i, m in enumerate(matches):

        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    H, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    h, w = template[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned
