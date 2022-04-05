import cv2
import sys
import ocr
import numpy as np
import matplotlib.pyplot as plt

def display(window, img, width, block=False):
    """Display an opencv image.
    """
    cv2.namedWindow(window, 
                    cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 
                     width, 
                     (int)(width * img.shape[0] / img.shape[1]))
    cv2.imshow(window, 
               img)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def close_op(img, size):
    """Morphological closing operation.
    """
    pre_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ero_element = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                            (size, (int)(size / 8)))
    dil_element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (size, (int)(size / 6)))
    pre = cv2.erode(img, 
                    pre_element, 
                    iterations = 1)
    dil = cv2.dilate(pre, 
                     dil_element, 
                     iterations = 1)
    ero = cv2.erode(dil, 
                    ero_element, 
                    iterations = 1)
    return ero

def close_op_y(img, size):
    ero_element = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                            ((int)(size / 8), size))
    dil_element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            ((int)(size / 6), size))
    dil = cv2.dilate(img, 
                     dil_element, 
                     iterations = 1)
    ero = cv2.erode(dil, 
                    ero_element, 
                    iterations = 1)
    return ero

def check_plate(img, rect):
    """Check for license plates. Not implemented.
    """
    return True

def draw_rect(img, rect):
    """Draws a rectangle in the image.
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return cv2.polylines(img, [box], 
                         isClosed=True, 
                         color=(0, 0, 255), 
                         thickness=1)

def scale_rect(rect, scale = 1.0):
    """Scale the rectangle area.
    """
    rect = list(rect)
    scale = scale * rect[1][0] - rect[1][0]
    rect[1] = [x + scale for x in rect[1]]
    return rect

def getBlueChannel(img):
    """Get the blue and green channels of the image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([40, 88, 40])
    upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    _, s, _ = cv2.split(hsv)
    img = cv2.bitwise_and(s, s, mask=mask)
    return img

def adjust_box(img, box):
    """Adjust the license plate area.
    """
    pts = np.array(np.where(img > 0)).T
    pts[:, [0,1]] = pts[:, [1, 0]]
    new_box = []
    for corner in box:
        new_box.append(list(pts[np.argmin([np.linalg.norm(pt - corner) for pt in pts])]))
    new_box = np.array(new_box)
    new_box[0][0] -= 3
    new_box[3][0] -= 3
    return new_box
    
def box_normalize(box):
    box = box[np.argsort(box[:,1])]
    if box[0][0] > box[1][0]:
        box[[0,1], :] = box[[1,0], :]
    if box[2][0] < box[3][0]:
        box[[2,3], :] = box[[3,2], :]
    return box

def extract_plate(src, rect):
    """Extract the license plate from the rectangular area.
    """
    rect = scale_rect(rect, 1.2)
    
    """Crop out the rectangular area"""
    img = np.zeros_like(src).astype('uint8')
    box = np.int0(cv2.boxPoints(rect))
    box = box_normalize(box)
    img = cv2.fillPoly(img, [box], (255, 255, 255))
    cv2.copyTo(src, img, img)
    
    """Some preprocessing operations"""
    img = getBlueChannel(img)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    th = (np.mean(img) + (np.max(img) - np.mean(img)) * 0.4)
    _, img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    img = close_op(img, 32)

    """Adjust the plates area"""
    cv2.polylines(src, [box], isClosed=True, color=(0, 0, 255), thickness=1)
    box = adjust_box(img, box)
    # cv2.polylines(src, [box], isClosed=True, color=(0, 0, 255), thickness=1)

    """Correct the distortion by perspective transformation"""
    box = box.astype('float32')

    M = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [179, 0], [179, 59], [0, 59]]))
    plate = cv2.warpPerspective(src, M, (180, 60))

    return plate

def plot_data(data):
    """Plot data"""
    x = np.linspace(0, len(data), len(data))
    plt.plot(x, data)
    plt.show()

def calc_potential(img):
    """Calculate the potential energy.
    """
    sum = np.sum(img, 0)
    pot = np.zeros_like(sum)
    p, rp = 0, 0
    for i in range(len(sum)):
        """Linearly increase exponential decay"""
        p = p + sum[i] if sum[i] else p / 5
        rp = rp + sum[len(sum) - i - 1] if sum[len(sum) - i - 1] else rp / 5
        pot[i] += p
        pot[len(sum) - i - 1] += rp
    # plot_data(pot)
    return pot

def adjust_vision(img):
    """Adjust the field of view size to better match.
    """
    # display('ch', img, 100, block=True)
    sum = np.sum(img, 1) 
    up, down = -1, -1
    for i in range(len(sum)):
        if sum[i] > 512:
            if up == -1:
                up = i
            down = i
    height = down - up + 1
    img = img[up: down+1, :]
    img = cv2.resize(img, ((int)(img.shape[1] * 1.2), height))
    padding = height - img.shape[1]
    left = (int)(padding  / 2)
    right = (int)(padding - left)
    img = np.hstack((np.zeros((height, left)), img))
    img = np.hstack((img, np.zeros((height, right))))
    img = cv2.resize(img, (20, 20))
    return img

def plate_split(img):
    """Split characters by potential energy.
    """
    pot = calc_potential(img)
    limitpot = np.mean(pot) * 0.5  
    bound = []
    left = 0
    for i in range(len(pot) - 1):
        if pot[i] < limitpot and pot[i + 1] >= limitpot:
            left = i
        elif pot[i] >= limitpot and pot[i + 1] < limitpot:
            bound.append([left, i])
            left = 0
    if left != 0:
        bound.append([left, len(pot - 1)])
    return bound

def plate_cut_text(img):
    sum = np.sum(img, 1)
    limit = np.mean(sum) * 0.2
    bound = []
    start = 0
    for i in range(len(sum) - 1):
        if sum[i] < limit and sum[i + 1] >= limit:
            start = i
        elif sum[i] >= limit and sum[i + 1] < limit:
            bound.append([start, i])
            start = 0
    if start != 0:
        bound.append([start, len(sum - 1)])
    up, down = 0, 0
    for b in bound:
        if b[1] - b[0] > down - up:
            up = b[0]
            down = b[1]

    return img[up: down + 1, :]

reader = ocr.Reader()
def plate_recognition(plate):
    """Core license plate recognition algorithm.
    """
    img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    th = (np.mean(img) + (np.max(img) - np.mean(img)) * 0.2)
    _, img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    img = plate_cut_text(img)
    display('plate th', img, 360)
    bound = plate_split(img)
    plate_res = []
    for b in bound:
        if b[1] - b[0] > 5 and b[1] - b[0] < 28:
            if len(plate_res) == 0:
                ch = reader.recognize_chinese(adjust_vision(img[:, b[0]:b[1]+1]))[0]
            else:
                ch = reader.recognize_alnum(adjust_vision(img[:, b[0]:b[1]+1]))[0]
            plate_res.append(ch)
    return plate_res

if __name__ == '__main__':
    """Get the source image"""
    if len(sys.argv) > 1:
        src = cv2.imread('./test/' + sys.argv[1])
    else:
        src = cv2.imread('./test/test7.jpg')

    """Extract the blue channel"""
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lower = np.array([100, 90, 40])
    upper = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    img = cv2.bitwise_and(s, s, mask=mask)

    """Preprocess"""
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    """Detect the texture in the image"""
    flipped = cv2.flip(img, 1)
    sobel1 = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 3)
    sobel2 = cv2.flip(cv2.Sobel(flipped, cv2.CV_8U, 1, 0, ksize = 3), 1)
    img = sobel1 / 2 + sobel2 / 2
    img = img.astype('uint8')


    """Thresholding"""
    th = (np.mean(img) + (np.max(img) - np.mean(img)) * 0.6)
    ret, img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)

    """Morphological closed operation"""
    img = close_op(img, 36)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > 50000:
            continue

        rect = cv2.minAreaRect(contour)
        if rect[1][0] * rect[1][1] < 666 or area < rect[1][0] * rect[1][1] * 0.6:
            continue

        dy, dx = contour.flatten().reshape(contour.shape[0],-1).T.ptp(1)
        cwb = rect[1][1] / rect[1][0] if rect[1][1] > rect[1][0] else rect[1][0] / rect[1][1]
        if dy < dx or cwb < 2.5 or cwb > 6:
            continue

        if not check_plate(src, rect):
            continue

        plate = extract_plate(src, rect)
        display('plate', plate, 360)

        plate_res = plate_recognition(plate)
        print(plate_res)

    display('res', src, 720, block=True)    

