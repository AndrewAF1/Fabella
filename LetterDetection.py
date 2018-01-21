import cv2
import numpy as np



def seperate(txtFile):
    im = cv2.imread(txtFile, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(im,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bg_sub = cv2.createBackgroundSubtractorMOG2()
    no_bg = bg_sub.apply(image)

    #POTENTIALLY IMPORTANT THING
    img = cv2.drawContours(no_bg, contours, -1, (0, 0, 0), 3, -1)
    #img = no_bg

    chars = []
    for con in range(1, len(contours)):
        #mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
        #cv2.drawContours(mask, contours, con, 255, -1) # Draw filled contour in mask. Third arg is num of contour
        #out = np.zeros_like(img) # Extract out the object and place into output image
        #out[mask == 255] = img[mask == 255]

        x, y, w, h = cv2.boundingRect(contours[con])
        out = cv2.rectangle(img, (x-30, y-30), (x + 30 + w, y + 30 + h), (0, 255, 0), 2)
        out = out[y-15:y+h+15, x-15:x+w+15]
        out = bg_sub.apply(out)

        (thresh, out) = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        chars.append(out)
    #print(chars)
    return(chars)


#cv2.waitKey()
