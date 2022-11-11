import time

import cv2
import numpy as np
def takeSecond(elem):
    return elem[0]+ elem[1]
# 为每 一个像素选择一个合适数目的高斯分布
if __name__ == '__main__':
    count = 10
    cap = cv2.VideoCapture("test.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    subtractor = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=3844, detectShadows=False)
    # subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    # subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=5, decisionThreshold=150)
    FILE_OUTPUT = '01.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 30, (width,height))
    while True:
        count = count + 1
        _, frame = cap.read()
        start = time.time()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dst = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        image = np.array(dst)

        mask = subtractor.apply(image)
        key = cv2.waitKey(30)
        if key == 27:
            break
        _, thresh = cv2.threshold(mask, 100, 150, cv2.THRESH_BINARY)
        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            # finding area of contour
            area = cv2.contourArea(contour)
            # if area greater than the specified value the only then we will consider it
            # find the rectangle co-ordinates
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x,y, w, h])


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, 'MOTION DETECTED', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            end = time.time()
            print(end-start)
            cv2.imshow('frame', frame)
            cv2.imshow('frame2', thresh)
            out.write(frame)
        # cv2.imwrite(str(count) + '.jpg', mask)
        cv2.imwrite(str(count) + 'MOG2.png', frame)
        cv2.imwrite(str(count) + 'nMOG2.png', mask)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
