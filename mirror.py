import cv2 as cv

image = cv.imread('sprites/ken/idle/0.png', flags=cv.IMREAD_UNCHANGED)
cv.imshow('1', image)
image = cv.flip(image, 1)
cv.imshow('mirrored', image)
cv.waitKey(0)
