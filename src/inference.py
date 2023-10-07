from model import ResNetLoader
from preprocess import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
from sklearn.svm import SVC
import pickle
import numpy as np

class Predictor:


    def __init__(self) -> None:
        pass

    
    def predict(self, resnet_model, svc_clf, image):

        resnet_feats = resnet_model.predict(image)
        preds = svc_clf.predict(resnet_feats[0])

        return preds

    

    def evaluate_model(self, resnet_model, svc_clf, x_test, y_test):

        # iterate over the x_test dataframe to extract the features with Resnet model
        cnn_feats_test = []
        for img in x_test:
            img = np.reshape(img,[1,IMAGE_SIZE,IMAGE_SIZE,3]) # reshape to fit cnn model input
            img_cnn_feats = resnet_model.predict(img) # input to cnn to extract features
            cnn_feats_test.append(img_cnn_feats[0])

        preds = svc_clf.predict(cnn_feats_test)
        acc = accuracy_score(y_test, preds)
        print("Model Test Accuracy: ", acc)
        svm_f1_score = f1_score(y_test, preds, average='micro')
        print("SVM Model F1 Score: ", svm_f1_score)
        print(classification_report(y_test, preds))




if __name__ == '__main__':


    IMAGE_SIZE = 224
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    EPOCHS = 100


    
    output_dir = "src/model_weights/"

    # load the training and testing datasets
    data_dir = 'data/'
    data_path = data_dir + 'data.csv'
    dataloader = DataLoader(data_path)

    x_train, x_test, y_train, y_test = dataloader.preprocess_data(data_dir, IMAGE_SIZE)
    print("Data loaded correctly.")

    resnet_loader = ResNetLoader(IMAGE_SIZE, NUM_CLASSES)        

    resnet_extractor = resnet_loader.get_ResNet18_extractor()


    predictor = Predictor()

    print(x_train[:1].shape)

    # select random image index from the test dataset.
    rand_id = random.sample(range(0, x_test.shape[1]-1), 1)
    rand_test_img = x_test[rand_id, :]
    
    
    svc_clf = pickle.load(open(output_dir + 'svc.p', 'rb'))
    img_label = predictor.predict(resnet_extractor, svc_clf, rand_test_img)

    print("Image True Label: ", y_test[rand_id], " Predicted Label: " ,img_label)

    # uncomment to display the model's performance on both the training and testing datasets. 
    print("Model Results on Test Data: ")
    predictor.evaluate_model(resnet_extractor, svc_clf, x_test, y_test)

