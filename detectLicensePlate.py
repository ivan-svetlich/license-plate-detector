import numpy as np
import cv2
import imutils
import pytesseract
import re 

def detect(image):
  pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

  # Regular expression for Argentinian license plates
  # License plates in Argentina are 'ABC 123' (old) or 'AB 123 CD' (new)
  reg = r"([A-Z]{2}[ \t]+[0-9]{3}[ \t]+[A-Z]{2}|[A-Z]{3}[ \t]*[0-9]{3})" 

  # Convert frame to gray scale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Convolve with Gaussian filter
  gray = cv2.GaussianBlur(gray, (3, 3), 0)

  # Apply Canny Edge Detection algorithm
  edged = cv2.Canny(image=gray, threshold1=100, threshold2=200)

  # Retrieve contours from the binary image
  cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # Select the 5 contours with the biggest area
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

  screenCnt = []
  valid_plate = None

  # For each selected contour
  for c in cnts:
    # Check if it has a rectangular shape
    perimeter = cv2.arcLength(c, True)  
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4: 
      # Search for place numbers comparing with the previously defined regex
      x,y,w,h = cv2.boundingRect(c) 
      new_img=image[y:y+h,x:x+w]
      plate = pytesseract.image_to_string(new_img, lang='eng')
      valid_plate = re.search(reg, plate)
      if valid_plate:
        screenCnt = approx
        print("Number plate is:", valid_plate.group(1))
      break

  # If screenCnt is empty (no valid plates found in frame), return None
  if len(screenCnt) == 0:
    return (None, '')

  # Else, return the contour and the plate number
  return (screenCnt, valid_plate.group(1))