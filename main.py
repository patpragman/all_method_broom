from paper_models.logreg import get_best_model_logreg
from paper_models.svm import get_best_model_svm
from paper_models.resnet18 import get_best_resnet
from paper_models.alexnet import get_best_AlexNet
from paper_models.cnn import get_best_artisanal_cnn
from paper_models.mlp import get_best_mlp

if __name__ == "__main__":
    get_best_mlp()
    get_best_AlexNet()
    get_best_resnet()
    get_best_artisanal_cnn()
    get_best_model_svm()
    get_best_model_logreg()
