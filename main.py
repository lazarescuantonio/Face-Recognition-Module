import os

# OpenCV; Library of functions mainly aimed at real-time computer vision
import cv2

# Library for mathematical operations and manipulation of data in the form of arrays
import numpy as np

# MobileNetV3Small model from TensorFlow's Keras applications module
# MobileNetV3 is a pre-trained Convolutional Neural Network (CNN) for feature extraction from images
# Small variant is a lighter version with fewer parameters and reduced computational requirements
from tensorflow.keras.applications import MobileNetV3Small

# Function to preprocess images to meet MobileNetV3Small requirements
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Modules for image processing with TensorFlow
from tensorflow.keras.preprocessing import image

# K-Nearest Neighbors (KNN) classifier
# KNN is a simple, easy-to-implement supervised machine learning algorithm
# KNN can be used to solve both classification and regression problems
from sklearn.neighbors import KNeighborsClassifier

# Function used to calculate the accuracy of a model by comparing predicted and actual values
from sklearn.metrics import accuracy_score

# Library used to display graphs and images
import matplotlib.pyplot as plt




# preprocess_image prepares an input_image for feeding it into MobileNetV3 model
def preprocess_image(input_image):
    # converts input_image to a numeric array format suitable for CNN processing
    array_image = image.img_to_array(input_image)
    
    # adds an extra dimension to the image tensor to
    # make it compatible with the model's batch processing requirements
    batch_image = np.expand_dims(array_image, axis = 0)
    
    # MobileNetV3 expects input in the range [-1, 1] (normalized input)
    normalized_image = preprocess_input(batch_image)
    return normalized_image


# weights     = "imagenet"    - loaded with pre-trained weights from the ImageNet dataset
# include_top = False         - removes the top classification layers, making the model adaptable for custom tasks
# input_shape = (224, 224, 3) - sets the expected input size to 224x224 pixels with 3 color channels (RGB)
base_model = MobileNetV3Small(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))


# extract features using MobileNetV3
def extract_features(detected_face):
    # preprocess_image prepares a detected_face for feeding it into MobileNetV3 model
    normalized_detected_face = preprocess_image(detected_face)
    
    # uses the base_model to extract_features from the normalized_detected_face
    extracted_features = base_model.predict(normalized_detected_face)
    
    # reshapes the extracted_features into a two-dimensional array; shape is adjusted so that
    # one dimension is the number of images (assuming extracted_features.shape[0] is the batch size)
    # and the other is a flattened feature vector (indicated by -1, which lets NumPy automatically calculate the size)
    flattened_features = np.reshape(extracted_features, (extracted_features.shape[0], -1))
    return flattened_features


# detect_face takes an input_image, detects face in it and returns the cropped and resized first detected_face
def detect_face(input_image):
    # input_image is converted to a gray_image
    # color information is usually not needed in face detection and grayscale simplifies the computation
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # loads a Pre-trained Haar Cascade classifier from OpenCV's library to detect_faces (popular choice)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # detectMultiScale is used to detect faces in the gray_image
    # returns a list of rectangles where faces are detected
    detected_faces = haar_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
    
    # if at least one face is detected then:
    if len(detected_faces) > 0:
        # extracts the coordinates of the first face
        x, y, w, h = detected_faces[0]
        
        # crops this region from the input_image (original image)
        detected_face = input_image[y:y + h, x:x + w]
        
        # resizes it to 224x224 pixels, which is the required input size for the MobileNetV3 model
        detected_face = cv2.resize(detected_face, (224, 224))
        return detected_face
    # if no faces are detected then:
    else:
        return None


# display an input_image alongside its detected_face
def display_face(input_image, detected_face):
    plt.figure(figsize = (8, 8))

    # display the input_image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # display the detected_face
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
    plt.title("Detected Face")
    plt.axis("off")
    
    plt.show()


def detect_name(predicted_label):
    # mapping between labels and personality names
    personality = {
        0: "audrey_tautou"  ,
        1: "emmanuel_macron",
        2: "eva_green"      ,
        3: "kylian_mbappe"  ,
        4: "marion_cotillard"
    }
    # returns the corresponding personality name for the predicted label
    
    # if the predicted_label is     found in the personality dictionary, returns the associated name
    # if the predicted_label is NOT found in the personality dictionary, returns "Unknown"
    return personality.get(predicted_label, "Unknown")


personalities_features = []
labels                 = []

# os.path.join(path_directory) - returns the absolute path to the "path_directory"
# os.listdir                   - returns a list of personalities names from the directory
# sorted                       - sort this list of personalities names in alphabetical order
# enumerate                    - iterates through the names and positions of the personalities in the list

path_directory = "./reference"

personalities = sorted(os.listdir(os.path.join(path_directory)))

for label, personality in enumerate(personalities):
    path_personality = os.path.join(path_directory, personality)
    images           = os.listdir(path_personality)
    
    print()
    print("Personality: ", personality)
    print("Label      : ", label      )
    print()
    
    for i in range(len(images)):
        path_image   = os.path.join(path_personality, images[i])
        input_image  = cv2.imread(path_image) # read an image file from the "path_image" using OpenCV
        
        # get the cropped and resized first detected_face
        detected_face = detect_face(input_image)
        
        if detected_face is not None:
            # display_face(input_image, detected_face) # FIXME debug
            
            # extract features using MobileNetV3
            flattened_features = extract_features(detected_face).flatten()
            personalities_features.append(flattened_features)
            labels.append(label)


# converts the lists into NumPy arrays to use to train the KNN classifier
personalities_features_array = np.array(personalities_features)
labels_array                 = np.array(labels)


# a simple KNN classifier is trained
# the classifier will use the 5 nearest neighbors to make predictions
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
# fit method trains the classifier using the dataset
knn_classifier.fit(personalities_features_array, labels_array)


# face recognition
def predict_face(test_image):
    # get the cropped and resized first detected_face
    detected_face = detect_face(test_image)
    
    # if a face is detected then:
    if detected_face is not None:
        # display_face(test_image, detected_face) # FIXME debug
        
        # extract features using MobileNetV3
        flattened_features = extract_features(detected_face).flatten()
        
        # uses the trained knn_classifier to find the 3 closest neighbors
        # knn_classifier returns the distances and indices (positions in the training set) of these neighbors
        distances, indices = knn_classifier.kneighbors(flattened_features.reshape(1, -1), n_neighbors = 3)
        
        # knn_classifier predicts the label of the test_image based on its features
        predicted_labels = knn_classifier.predict(flattened_features.reshape(1, -1))
        
        return predicted_labels, indices.flatten(), distances.flatten()
    # if no face is detected then:
    else:
        return None, None, None


distances_list        = []
indices_list          = []
actual_labels_list    = []
predicted_labels_list = []


# os.path.join(path_directory) - returns the absolute path to the "path_directory"
# os.listdir                   - returns a list of personalities names from the directory
# sorted                       - sort this list of personalities names in alphabetical order
# enumerate                    - iterates through the names and positions of the personalities in the list

# the face recognition system is tested for each image in the "test" directory
path_directory = "./test"

personalities = sorted(os.listdir(os.path.join(path_directory)))

for label, personality in enumerate(personalities):
    path_personality = os.path.join(path_directory, personality)
    images           = os.listdir(path_personality)
    
    print()
    print("Personality: ", personality)
    print("Label      : ", label      )
    print()
    
    for i in range(len(images)):
        path_image   = os.path.join(path_personality, images[i])
        input_image  = cv2.imread(path_image) # read an image file from the "path_image" using OpenCV
        
        # the predicted_labels, indices and distances are obtained
        predicted_labels, indices, distances = predict_face(input_image)
        
        if predicted_labels is not None:
            personality_name =  detect_name(predicted_labels[0])
            reference_names  = [detect_name(labels_array[i]) for i in indices]
            
            print()
            print("Actual    Personality:", personality        )
            print("Acutal    Label      :", label              )
            print("Predicted Personality:", personality_name   )
            print("Predicted Label      :", predicted_labels[0])
            print("Predicted Labels     :", predicted_labels   )
            print("Closest   References :", reference_names    )
            print("Distances            :", distances[:3]      )
            print()
            
            display_face(input_image, detect_face(input_image)) # FIXME debug
            
            distances_list.append(distances[:3])
            indices_list.append(indices)
            actual_labels_list.append(label)
            predicted_labels_list.append(predicted_labels[0])

# converts the lists into NumPy arrays to compute the accuracy
actual_labels_array    = np.array(actual_labels_list)
predicted_labels_array = np.array(predicted_labels_list)


# compute accuracy
accuracy = accuracy_score(actual_labels_array, predicted_labels_array)
print()
print("====================")
print("Accuracy:", accuracy)
print("====================")
print()


# display the first 3 distances and their corresponding names for each prediction
for i in range(len(actual_labels_list)):
    print()
    print(f"Image {i+1}")
    print("Actual    Personality:", detect_name(actual_labels_list[i]))
    print("Predicted Personality:", detect_name(predicted_labels_list[i]))
    print("Closest   References :", [detect_name(labels_array[j]) for j in indices_list[i]])
    print("Distances            :", distances_list[i])
    print()




# display the data set
def display_images(path_directory):
    # os.path.join(path_directory) - returns the absolute path to the "path_directory"
    # os.listdir                   - returns a list of personalities names from the directory
    # sorted                       - sort this list of personalities names in alphabetical order
    # enumerate                    - iterates through the names and positions of the personalities in the list
    
    personalities = sorted(os.listdir(os.path.join(path_directory)))
    
    plt.figure(figsize = (1920/100, 1080/100), dpi = 100) # screen resolution
    
    for label, personality in enumerate(personalities):
        path_personality = os.path.join(path_directory, personality)
        images           = os.listdir(path_personality)
        
        for i in range(len(images)):
            path_image   = os.path.join(path_personality, images[i])
            input_image  = cv2.imread(path_image) # read an image file from the "path_image" using OpenCV
            
            plt.subplot(len(personalities), len(images), label * len(images) + i + 1)
            plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            plt.axis("off") # removes the axes to center the image
            plt.title(f"{personality}")
    plt.savefig(path_directory)
    plt.show()


display_images(path_directory = "./reference")
display_images(path_directory = "./test")

