import numpy as np
import cv2
from pyzbar.pyzbar import decode

def colorFragmentation(img_color, imgraw):
    # Konwersja z przestrzeni bgr do hsv
    imageRaw = np.copy(imgraw)
    imageFrame = np.copy(img_color)
    imageTrueFrameRoI = np.copy(img_color)
    imageTrueFrame = np.copy(img_color)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Ustawienie przedziałów wartości kolorów dla masek
    #złożenie maski czerwonej ze względu na to że w hsv wypada na granicy wartości
    red_lower = np.array([170, 110, 110], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    red_lower = np.array([0, 120, 120], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)
    red_mask2 = cv2.inRange(hsvFrame, red_lower, red_upper)
    red_mask = cv2.bitwise_or(red_mask, red_mask2)

    green_lower = np.array([45, 80, 80], np.uint8)
    green_upper = np.array([85, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([90, 80, 80], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Poprawienie masek filtrami morfologicznymi (uzupełnienie luk i dokładniejsze objęcie obszaru)

    kernal = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    green_mask = cv2.dilate(green_mask, kernal)
    blue_mask = cv2.dilate(blue_mask, kernal)

    # Stworzenie konturów dla poszczególnych masek wraz z wyborem tego o największej powierzchnii
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    Biggest_area = {"color": 'None', "area": 0, "rectangle": [0, 0, 0, 0]}
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 225):
            x, y, w, h = cv2.boundingRect(contour)
            if area > Biggest_area["area"]:
                Biggest_area["area"] = area
                Biggest_area["color"] = "red"
                Biggest_area["rectangle"] = [x, y, w, h]
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))


    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 225):
            x, y, w, h = cv2.boundingRect(contour)
            if area > Biggest_area["area"]:
                Biggest_area["area"] = area
                Biggest_area["color"] = "green"
                Biggest_area["rectangle"] = [x, y, w, h]
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(imageFrame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))

    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 225):
            x, y, w, h = cv2.boundingRect(contour)
            if area > Biggest_area["area"]:
                Biggest_area["area"] = area
                Biggest_area["color"] = "blue"
                Biggest_area["rectangle"] = [x, y, w, h]
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

    # wydobycie informacji o największym obszarze koloru i jego kolorze
    color = Biggest_area["color"]
    x, y, w, h = Biggest_area["rectangle"]

    imageTrueFrameRoI = cv2.rectangle(imageTrueFrameRoI, (x, y),
                               (x + w, y + h),
                               (255, 0, 255), 2)

    cv2.putText(imageTrueFrameRoI, color, (x+w, y-20+h),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 255))

    cv2.imshow("Color Detection in Real-TIme", imageFrame)
    cv2.imshow("Biggest color region", imageTrueFrameRoI)

    #imageRoi = imageRaw[y:y+h, x:x+w]
    if h > 100 and w > 100: # w przypadku wykrycia odpowiednio dużego obszaru koloru
                            #do detekcji qr wykorzystywane jest ograniczony obraz, w przeciwnym razie cały

        roi_mask = imageRaw.copy()
        roi_mask[:, :] = 0
        roi_mask[y:y+h, x:x+w] = 255
        image_roi = cv2.bitwise_and(roi_mask, imageRaw)
        cv2.imshow("Region defined by biggest color cluster", image_roi)
        output = decoder(image_roi, imageRaw)
        output = cv2.rectangle(output, (x, y),
                               (x + w, y + h),
                               (255, 0, 255), 2)

        cv2.putText(output, color, (x + w, y - 20 + h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 255))
    else:
        output = decoder(imageTrueFrame, imageRaw)
        output = cv2.rectangle(output, (x, y),
                                   (x + w, y + h),
                                   (255, 0, 255), 2)
        cv2.putText(output, color, (x + w, y - 20 + h),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 255))

    return Biggest_area, imageFrame, output





def displaycontours(image_grayscale, image_raw):
    #znajduje i wyświetla kontury wraz z ramkami ograniczającymi
    dst = image_raw.copy()
    image = image_grayscale.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 400:
            dst = cv2.drawContours(dst, [cnt], -1, (0, 0, 255), 2)
            dst = cv2.rectangle(dst, (x, y),
                                (x + w, y + h),
                                (255, 0, 255), 2)
    return dst

def displaybiggestcontour(image_grayscale, image_raw):
    #Znajduje i wyświetla największy kontur i ramkę ograniczającą
    dst = image_raw.copy()
    image = image_grayscale.copy()
    Biggest_area = {"area": 0, "rectangle": [0, 0, 0, 0]}
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:

        Biggest_area = {"area": 0, "rectangle": [0, 0, 0, 0], "contour": contours[0]}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if area > Biggest_area["area"]:
                Biggest_area["area"] = area
                Biggest_area["rectangle"] = [x, y, w, h]
                Biggest_area["contour"] = cnt
        x, y, w, h = Biggest_area["rectangle"]
        cnt = Biggest_area["contour"]
        dst = cv2.drawContours(dst, [cnt], -1, (0, 0, 255), 2)
        dst = cv2.rectangle(dst, (x, y),
                               (x + w, y + h),
                               (255, 0, 255), 2)
    rect = Biggest_area["rectangle"]
    return dst, rect

def displaycontoursrectangle(imageThresh, imageRaw):
    #zaznaczenie na obrazie prostokątów
    dstt = imageThresh.copy()
    dstr = imageRaw.copy()
    contours, hierarchy = cv2.findContours(dstt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if (len(approx) < 5) & (len(approx) > 3):
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if h > 10 and w > 10 and (ratio < 0.9 or ratio > 1.1):
                cv2.putText(dstr, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                dstr = cv2.drawContours(dstr, [cnt], -1, (0, 0, 255), 2)
    return dstr

def decoder(img, img_raw):
    #wykrywa i oznacza qr_code
    image = img.copy()
    image_raw = img_raw
    gray_img = cv2.cvtColor(image, 0)
    barcode = decode(gray_img)

    for obj in barcode:
        points = obj.polygon
        x, y, w, h = obj.rect
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_raw, [pts], True, (0, 255, 0), 3)
        cv2.putText(image_raw, "QR Code", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return image_raw