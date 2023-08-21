import cv2
import APOFunctions

# Przygotowanie streamu z kamery
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Inicjalizacja klatek k-1 i k-2
ret, image_00 = vid.read()
ret, image_01 = vid.read()

# konwersja z przestrzeni barw bgr do skali szarości
image_00 = cv2.cvtColor(image_00, cv2.COLOR_BGR2GRAY)
image_01 = cv2.cvtColor(image_01, cv2.COLOR_BGR2GRAY)
roi_pos = [0, 0, 1280, 720];

while True:
    # Przygotowanie jąder do operacji morfologicznych
    kernelsize_dialate1 = 1;
    kernelsize_dialate2 = 8;
    kernelsize_erode1 = 1;
    kernelsize_erode2 = 2;

    kernel_dialate1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelsize_dialate1 + 1, 2 * kernelsize_dialate1 + 1))
    kernel_dialate2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernelsize_dialate2 + 1, 2 * kernelsize_dialate2 + 1))
    kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * kernelsize_erode1 + 1, 2 * kernelsize_erode1 + 1))
    kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * kernelsize_erode2 + 1, 2 * kernelsize_erode2 + 1))

    # Pobranie klatki k i konwersja do skali szarości
    ret, frame = vid.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Obliczenie różnicy bezwzględnej
    img_diff0 = cv2.absdiff(image_00, image_01)
    img_diff1 = cv2.absdiff(image_01, gray_frame)
    cv2.imshow('2 frame diff - grayscale', img_diff1)

    img_diff0 = cv2.threshold(img_diff0, 15, 255, cv2.THRESH_BINARY)[1]
    img_diff1 = cv2.threshold(img_diff1, 15, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('2 frame diff - binary', img_diff0) #

    #operacje morfologiczne:
    img_diff0_er = cv2.erode(img_diff0, kernel_erode1, iterations=1)
    img_diff0_er = cv2.dilate(img_diff0_er, kernel_dialate1, iterations=1)
    #cv2.imshow('2 frame diff - or - opened', img_diff0_er) #

    img_diff1_er = cv2.erode(img_diff1, kernel_erode1, iterations=1)
    img_diff1_er = cv2.dilate(img_diff1_er, kernel_dialate1, iterations=1)

    #Połączenie klatek w obraz wyjściowy
    #dla zobrazowania, wynik bez otwarcia
    image_diff_final = cv2.bitwise_or(img_diff0, img_diff1)
    image_diff_final = cv2.threshold(image_diff_final, 80, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('3 frame diff - or - no opening', image_diff_final) #
    #zastosowane ostatecznie, lub z wcześniejszym otwarciem
    image_diff_final_er = cv2.bitwise_or(img_diff0_er, img_diff1_er)
    #cv2.imshow('3 frame diff - before closing', image_diff_final_er) #

    #operacje morfologiczne do poprawienie konturu i usunięcia szumów

    image_diff_final_erdi = cv2.dilate(image_diff_final_er, kernel_dialate2, iterations=1)
    cv2.imshow('3frame diff - eroded - dialated', image_diff_final_erdi) #

    image_diff_final = cv2.erode(image_diff_final_erdi, kernel_erode2, iterations=1)
    cv2.imshow('3 frame diff - closed', image_diff_final)

    #Wyznaczenie największego konturu i RoI
    eroded_cnt, roirect = APOFunctions.displaybiggestcontour(image_diff_final, frame)
    cv2.imshow('3 frame diff - contour', eroded_cnt)

    # Warunek minimalnego pola konturu (w celu uniknięcia wyboru złego obiektu w przypadku małych ruchów)
    if roirect[2] > 300 and roirect[3] > 150:
        roi_pos = roirect

    roi_mask = frame.copy()
    roi_mask[:, :] = 0
    roi_mask[roi_pos[1]:roi_pos[1]+roi_pos[3], roi_pos[0]:roi_pos[0]+roi_pos[2]] = 255
    image_roi = cv2.bitwise_and(roi_mask, frame)
    image_roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    image = frame.copy()

    Biggest_area, image, output = APOFunctions.colorFragmentation(image_roi, image)
    cv2.imshow("Final Results - QR + Biggest color region", output)
    image_00 = image_01.copy()
    image_01 = gray_frame.copy()

    cv2.waitKey(100)

vid.release()
