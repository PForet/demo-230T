import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from mobilenet import MobileNetV2
from classifiers import ExpressionClassifier, GenderClassifier, AgeClassifier
from matplotlib import pyplot as plt 

class Model():
    def __init__(self):

        self.to_tensors = transforms.Compose([
            transforms.Resize((224,224)), # Resize to 224x224 pixels
            transforms.ToTensor(), # Transform the images into tensors 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.features_extractor = MobileNetV2()
        self.features_extractor.eval()

        expr_c_weights = 'models/params_expression_assembled.th'
        gender_c_weights = 'models/params_gender_classifier.th'
        age_c_weights = 'models/params_age_classifier.th'

        expression_c = ExpressionClassifier()
        expression_c.load_state_dict(torch.load(expr_c_weights, map_location='cpu'))

        gender_c = GenderClassifier()
        gender_c.load_state_dict(torch.load(gender_c_weights, map_location='cpu'))

        age_c = AgeClassifier()
        age_c.load_state_dict(torch.load(age_c_weights, map_location='cpu'))

        self.expression_c = expression_c
        self.gender_c = gender_c
        self.age_c = age_c
    
    def imgs_to_tensors(self, imgs):

        imgs = self.to_tensors(imgs)
        return imgs

    def classify(self, imgs):
        with torch.no_grad():
            tensors = self.imgs_to_tensors(imgs)
            tensors = tensors.view(-1,3,224,224)
            features = self.features_extractor(tensors)
            expressions = self.expression_c(features)
            gender = self.gender_c(features)
            age = self.age_c(features)
        return expressions, gender, age
