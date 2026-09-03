import cv2
import numpy as np

# Load lunar image
image = cv2.imread("moon.jpg")

if image is None:
    print("Error: moon.jpg not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce image noise
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect crater-like circular structures
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=25,
    param1=100,
    param2=35,
    minRadius=8,
    maxRadius=100
)

output = image.copy()
crater_count = 0

if circles is not None:
    circles = np.round(circles[0]).astype("int")

    for x, y, radius in circles:
        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        crater_count += 1

print("LUNAR CRATER DETECTION")
print("----------------------")
print("Detected crater-like structures:", crater_count)

cv2.imwrite("detected_craters.jpg", output)

print("Result saved as detected_craters.jpg")