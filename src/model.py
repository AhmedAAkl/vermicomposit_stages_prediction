from operator import mod
from statistics import mode
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
from tensorflow.keras import models
from classification_models.keras import Classifiers
from tensorflow.keras import Sequential


output_dir = "src/model_weights/"


class ResNetLoader:

    def __init__(self, IMAGE_SIZE, NUM_CLASSES) -> None:
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_CLASSES = NUM_CLASSES


    def get_ResNet18_model(self):
        
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18((224, 224, 3), include_top=False)
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))
        

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def get_ResNet18_extractor(self):
        
        resnet_model = self.get_ResNet18_model()
        print("ResNet model created correclty.")
        # load the model weights
        resnet_model.load_weights(output_dir + 'resnet_18_model.hdf5')
        # drop the last two layers
        resnet_model.pop()
        resnet_model.pop()
        # print(resnet_model.summary())

        return resnet_model   


    def model_summary(self):
        model = self.get_ResNet18_model()
        print("Model Summary: ")
        print(model.summary())




if __name__ == '__main__':

    NUM_CLASSES = 4
    IMAGE_SIZE = 224

    resnet_loader = ResNetLoader(IMAGE_SIZE, NUM_CLASSES)

    dense_model = resnet_loader.get_ResNet18_model()
    resnet_loader.get_ResNet18_extractor()

    # resnet_loader.model_summary()