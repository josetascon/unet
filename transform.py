import albumentations as alb
from albumentations.pytorch import ToTensorV2

def train_transform(pt = 0.75, pi = 0.5):
    # pt = total probability
    # pi = internal probability
    transform = alb.Compose( [
        alb.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.2,0.2)},p=pi),
        alb.Affine(scale=(0.8,1.2),p=pi),
        alb.Affine(rotate=(-5.0,5.0),p=pi),
        alb.Affine(shear=(-5.0,5.0),p=pi),
        ], p=pt)
    return transform

def validate_transform(pt = 1.0, pi = 0.5):
    # pt = total probability
    # pi = internal probability
    transform = alb.Compose( [
        alb.Normalize(mean=(0.5), std=(0.25)),
        alb.GaussNoise(var_limit=0.005, p=0.0),
        ToTensorV2(),
        ], p=pt)
    return transform

def predict_transform(pt = 1.0):
    # pt = total probability
    # pi = internal probability
    transform = alb.Compose( [
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], p=pt)
    return transform