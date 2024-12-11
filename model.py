from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL
from PIL import Image  # Import explicitly to avoid confusion


class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
    # Initialize lists for storing images and class labels
        img_list = []  # This will hold flattened images
        class_list = []  # This will hold class labels

    # Load images for class 1
        for i in range(1, counters[0]):  # Assuming counters[0] is the count of class 1 images
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Load in grayscale
            if img is not None:  # Ensure the image was loaded successfully
                img = img.reshape(-1)  # Flatten the image into a 1D array
                img_list.append(img)  # Add image to the list
                class_list.append(1)  # Append the label for class 1
            else:
                print(f"Warning: Failed to load image 1/frame{i}.jpg")

    # Load images for class 2
        for i in range(1, counters[1]):  # Assuming counters[1] is the count of class 2 images
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)  # Load in grayscale
            if img is not None:  # Ensure the image was loaded successfully
                img = img.reshape(-1)  # Flatten the image into a 1D array
                img_list.append(img)  # Add image to the list
                class_list.append(2)  # Append the label for class 2
            else:
                print(f"Warning: Failed to load image 2/frame{i}.jpg")

    # Convert lists to NumPy arrays
        if len(img_list) > 0:  # Check if we have data
            img_list = np.array(img_list)  # 2D array: shape (num_samples, flattened_image_size)
            class_list = np.array(class_list)  # 1D array: shape (num_samples,)

        # Train the model
            self.model.fit(img_list, class_list)
            print("Model successfully trained!")
        else:
            print("Error: No images were loaded. Training aborted.")


    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((50, 50), Image.Resampling.LANCZOS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')[:, :, 0]
        img = img.reshape(1900)
        prediction = self.model.predict([img])

        return prediction[0]