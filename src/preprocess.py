
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np




class DataLoader:

    def __init__(self, data_path):
        self.data_path = data_path

    def load_csv(self):
        # loading data
        return pd.read_csv(self.data_path)


    def resize_image(self, image, IMAGE_SIZE):
        return cv2.resize(image.copy(), IMAGE_SIZE, interpolation = cv2.INTER_AREA)


    def read_image(self, filepath):
        return cv2.imread(filepath)



    def preprocess_data(self, data_dir, IMAGE_SIZE):
        df = self.load_csv()
        X = np.zeros((df.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))

        for ind, row in df.iterrows(): 
            image = self.read_image(data_dir + row['path'])
            if image is not None:                
                X[ind] = self.resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
        X /= 255.        
        y = df['day_number'].values
        # splitting the data into test and training data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)
        return x_train, x_test, y_train, y_test






if __name__ == '__main__':
    
    data_dir = 'data/'
    data_path = data_dir + 'data.csv'

    IMAGE_SIZE = 224

    dataloader = DataLoader(data_path)

    x_train, x_test, y_train, y_test = dataloader.preprocess_data(data_dir, IMAGE_SIZE)
    print("Data loaded correctly.")


