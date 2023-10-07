from model import ResNetLoader
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from preprocess import DataLoader
import numpy as np
from sklearn.svm import SVC
import pickle

class Trainer:

    def __init__(self) -> None:
         pass
    


    def finetune_ResNet(self, model, x_train, y_train, EPOCHS, BATCH_SIZE, model_callbacks):



  
        try:
            model_history = model.fit(x_train, y_train, callbacks=model_callbacks, batch_size=BATCH_SIZE, 
                                    epochs=EPOCHS, verbose=1,validation_split=0.1)

        except KeyboardInterrupt:
                print("Execution interrputed by user.")
    
    def extract_features(self, model, data):
        # iterate over the data to extract the features with CNN model
        cnn_feats = []
        for img in data:
            img = np.reshape(img,[1,IMAGE_SIZE,IMAGE_SIZE,3]) # reshape to fit cnn model input
            img_cnn_feats = resnet_model.predict(img) # input to cnn to extract features
            cnn_feats.append(img_cnn_feats[0])
        
        return cnn_feats
             
         


if __name__ == '__main__':

    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 100


    
    output_dir = "src/model_weights/"


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    checkpoint_filepath = output_dir + 'class_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    densenet_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model_callbacks = [reduce_lr, early_stop, densenet_checkpoint_callback]


    # load the training and testing datasets
    data_dir = 'data/'
    data_path = data_dir + 'data.csv'
    dataloader = DataLoader(data_path)

    x_train, x_test, y_train, y_test = dataloader.preprocess_data(data_dir, IMAGE_SIZE)
    print("Data loaded correctly.")
    
    
    # The frist phase it to train ResNet-18 on the training data.
    resnet_loader = ResNetLoader(IMAGE_SIZE, NUM_CLASSES)        
    resnet_model = resnet_loader.get_ResNet18_model()
    print("ResNet-18 model created correclty.")


    trainer = Trainer()
    print("Start the training process...")
    trainer.finetune_ResNet(resnet_model, x_train, y_train, EPOCHS, BATCH_SIZE, model_callbacks)


    # use the finetuned ResNet-18 to extract features from the data and train SVC model to classify it.

    resnet_extractor = resnet_loader.get_ResNet18_extractor()
    # iterate over the x_train and y_train dataframes to extract the features with ResNet-18 model
    cnn_feats_train = trainer.extract_features(resnet_extractor, x_train)
    cnn_feats_test = trainer.extract_features(resnet_extractor, x_test)

    clf = SVC(C=1.0, kernel='rbf')
    clf.fit(cnn_feats_train, y_train)

    # saving the SVC model after training.
    pickle.dump(clf, open(output_dir + 'svc.p', 'wb'))


    