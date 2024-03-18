# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PcNeLZngM3Ymb9vNbRuoUdvI-0gWGIAG
"""

from google.colab import drive
import os
import cv2
import numpy as np
import pandas as pd

drive.mount('/content/drive')

os.chdir('/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)')

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/d'

# Initialize an empty DataFrame
df = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        # Resize the image
        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        df = df.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
df['Letter'] = "D"
print(df)

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/e'

# Initialize an empty DataFrame
sign_e = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_e = sign_e.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_e['Letter'] = "E"
print(sign_e)

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/h'

# Initialize an empty DataFrame
sign_h = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_h = sign_h.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_h['Letter'] = "H"
print(sign_h)

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/l'

# Initialize an empty DataFrame
sign_l = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_l = sign_l.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_l['Letter'] = "L"
print(sign_l)

folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/o'

# Initialize an empty DataFrame
sign_o = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_o = sign_o.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_o['Letter'] = "O"
print(sign_o)

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/r'

# Initialize an empty DataFrame
sign_r = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_r = sign_r.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_r['Letter'] = "R"
print(sign_r)

# Set the path/directory
folder_dir = '/content/drive/My Drive/PIC16B_PROJ/CollectedImages (1)/w'

# Initialize an empty DataFrame
sign_w = pd.DataFrame()

# Loop through the images in the directory
for filename in os.listdir(folder_dir):
    # Check if the file ends with .jpg
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_dir, filename)
        letter = cv2.imread(image_path)

        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a NumPy array
        np_array = np.array(letter)

        # Flatten the array
        flattened_array = np_array.flatten()

        # Append the flattened array to the DataFrame
        sign_w = sign_w.append(pd.Series(flattened_array), ignore_index=True)

# Display the DataFrame
sign_w['Letter'] = "W"
print(sign_w)

new = df.append(sign_e)
new = new.append(sign_h)
new = new.append(sign_l)
new = new.append(sign_o)
new = new.append(sign_r)
new = new.append(sign_w)

new.to_csv('sign_language_dataset.csv')

sign_language_url = "https://raw.githubusercontent.com/emilyrshi/PIC16B_Project_Repo/ad89b34572ad663eadcf5c88667f10ad97e791b9/SignLanguage_CSV/sign_language_dataset.csv"

sign_language_data = pd.read_csv(sign_language_url, index_col = False)
sign_language_data.head()