import cv2 as cv

image = cv.imread('../screenshots/ken_hadoken_projectile/20.png')

threshold = 255
canny_output = cv.Canny(image, threshold, threshold * 2)
contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_poly = [None] * len(contours)
bounding_boxes = [None] * len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    bounding_boxes[i] = cv.boundingRect(contours_poly[i])

for i in range(len(contours)):
    color = (255, 0, 0)
    cv.drawContours(image, contours_poly, i, color)
    # cv.rectangle(
    #     image,
    #     (int(bounding_boxes[i][0]),
    #     int(bounding_boxes[i][1])),
    #     (int(bounding_boxes[i][0] + bounding_boxes[i][2]),
    #     int(bounding_boxes[i][1] + bounding_boxes[i][3])),
    #     color,
    #     1
    # )

cv.imshow('', image)
cv.waitKey(0)
