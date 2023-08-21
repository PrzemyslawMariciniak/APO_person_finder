import APOFunctions
import cv2
import numpy as np

path = r'C:\Users\Robin\Documents\TEMPLATE - APO - BLUE - NEWER.png'
path_background = r'C:\Users\Robin\Documents\TEMPLATE - APO - BLANK.png'
# pobranie obrazów z dysku twardego
background = cv2.imread(path_background)
image = cv2.imread(path)
imageraw = cv2.imread(path)

img = np.copy(image)
image_width = np.size(image, 1)
image_height = np.size(image, 0)

cv2.imshow('Obraz wyjsciowy', image)

image_new = cv2.absdiff(image, background)

cv2.imshow('Roznica bezwzgledna', image_new)
image_new_grey = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
cv2.imshow('Skala szarosci', image_new_grey)

mask_threshold = cv2.threshold(image_new_grey, 0, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('maska_1', mask_threshold)
mask_threshold = cv2.multiply(mask_threshold, 1/255)
image2 = np.copy(image)
image2 = cv2.bitwise_and(image2, image2, mask=mask_threshold)
cv2.imshow('maska_2', image2)
image2_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image2_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Liczba wykrytych konturow:", len(contours))
goodCnt = []
for cnt in contours:
    x1, y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w)/h
        if (ratio < 0.9 or ratio > 1.1) and h > 10:
            cv2.putText(image, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            image = cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
            goodCnt.append([x, y, w, h])

            image = cv2.drawMarker(image, [x, y], (0, 255, 0), 2)
cv2.imshow("Ksztalty", image)
for cnt in goodCnt:
    print(cnt)
    x0 = cnt[0]
    y0 = cnt[1]

    roiArray = imageraw[cnt[1]:cnt[1]+cnt[3], cnt[0]:cnt[0]+cnt[2]]

    Biggest_area = APOFunctions.colorFragmentation(roiArray, imageraw)[0]
    color = Biggest_area["color"]
    x, y, w, h = Biggest_area["rectangle"]
    x = x + x0
    y = y + y0
    imagenew = cv2.rectangle(imageraw, (x, y),
                               (x + w, y + h),
                               (0, 255, 255), 2)

    cv2.putText(imagenew, color, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (160, 0, 200))


    cv2.imshow('Wykryte pole', imagenew)

    Biggest_area, image, output = APOFunctions.colorFragmentation(imagenew, image)
    cv2.imshow('Wykryte pole', output)
cv2.waitKey(0)
cv2.destroyAllWindows()