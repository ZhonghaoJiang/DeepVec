from DCGAN.dcgan import discriminator_model
from DCGAN.dcgan_svhn import discriminator_model_cifar10
import numpy as np

def picture_preprocess(images):
    image = np.asarray(images) / 255
    return image

class DCGAN:
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('DCGAN/discriminator_mnist')
        elif dataset == 'FM':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('DCGAN/discriminator_fm')
        elif dataset == 'SVHN':
            self.model = discriminator_model_cifar10()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('DCGAN/discriminator_svhn')

    def predict_batch(self, preprocessed_test_inputs):
        result = self.model.predict(preprocessed_test_inputs)
        return result

    def get_fidelity_list(self, test_input):
        test_images = test_input.reshape(-1, 28, 28, 1).astype(np.int16)
        preprocessed_input = picture_preprocess(test_images)
        fidelity_list = self.predict_batch(preprocessed_input)
        fidelity_list = np.squeeze(fidelity_list)
        sorted_indices = np.argsort(fidelity_list) # 降序排序
        return sorted_indices


