import cv2
import pytesseract



def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_noise(img):
    return cv2.medianBlur(img,5)

def thresholding(img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def dilatate(img):
    kernel = np.ones((5,5),np.unit8)
    return cv2.dilatate(img,kernel,iterations=1)

def erode(img):
    kernel = np.ones((5,5),np.unit8)
    return cv2.erode(img,kernel,iterations = 1)

def opening(img):
    kernel = np.ones((5,5),np.unit8)
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

def canny(img):
    return cv2.Canny(img,100,200)

def deskew(img):
    coords = np.column_stack(np.where(img>0))
    angle = cv2.minAeraRect(coords)[-1]
    if angle < -45:
        angle = -(90+angle)
    else:
        angle = -angle
    (h,w)=image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    rotated = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(img,template):
    return cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)

img = cv2.imread("WhatsApp-Image-2022-11-03-at-16.41.35.png")
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)