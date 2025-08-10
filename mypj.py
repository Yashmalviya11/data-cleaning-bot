import cv2

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

# Load an image (replace with your image path)
image = cv2.imread("test_image.jpg")

# Check if the image was loaded successfully
if image is not None:
    print("Image loaded successfully!")
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load image.")