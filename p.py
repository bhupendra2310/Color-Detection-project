import cv2
import numpy as np
from sklearn.cluster import KMeans

def color_detection(image_path, k=3):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Display the original image
    cv2.imshow("Original Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Create a blank image with the dominant colors
    color_chart = np.zeros((100, 300, 3), dtype=np.uint8)
    color_chart[:, :100] = dominant_colors[0]
    color_chart[:, 100:200] = dominant_colors[1]
    color_chart[:, 200:] = dominant_colors[2]

    # Display the color chart
    cv2.imshow("Dominant Colors", color_chart)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example: Detect color from an image
'''image_path = "‪C:/Users/lenovo/Desktop/20/colorpic.jpg"'''
image_path = "C:/Users/lenovo/Desktop/20/colorpic.jpg"

color_detection(image_path)
