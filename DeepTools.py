"""
This repo condense two interesting deep learning methods into one function for fun.

Made by Togel Bian in 18, February, 2021.
"""
from torchvision import models,transforms
from PIL import Image
from Network import ResNetGenerator,ResNetBlock
import torch
import cv2,time
import torch.nn.functional as f


class Recognize(object):
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.eval()
        with open('./data/imagenet_classes.txt', encoding='utf8') as f:
            self.labels = [line.split(',')[-1] for line in f.readlines()]

    def return_result(self,img):
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t,0)
        out = self.resnet(batch_t)
        percentage = f.softmax(out,dim=1)[0] * 100
        _,index = torch.max(out,1)
        result = (self.labels[index[0]],percentage[index[0]].item())
        return result

    def __call__(self,path):
        """
        :param path: give the image's absolute path or relative path(in current directory)
        :return: None but show the result
        """
        img = Image.open(path)
        result = self.return_result(img)
        print('Mostly like:{}by {}'.format(result[0],result[1]))


class CycleGAN(object):
    def __init__(self):
        self.netG = ResNetGenerator()

        model_path = './pkl/horse2zebra_0.4.0.pth'
        model_data = torch.load(model_path)
        self.netG.load_state_dict(model_data)
        self.netG.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

    def return_result(self,img):
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t,0)
        batch_out = self.netG(batch_t)
        out_t = (batch_out.data.squeeze() + 1.0) / 2.0
        out_img = transforms.ToPILImage()(out_t)
        return out_img

    def __call__(self, path):
        """
        :param path: give the image's absolute path or relative path(in current directory)
        :return: None but show the result
        """
        img = Image.open(path)
        out_img = self.return_result(img)
        out_img.show()


class SuperResolution(object):
    def __init__(self):
        model_name = modelName = "edsr"
        modelScale = 4
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("./models/EDSR_x4.pb")
        self.sr.setModel(modelName, modelScale)

    def __call__(self, path):
        image = cv2.imread(path)
        print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
        # use the super resolution model to upscale the image, timing how
        # long it takes
        start = time.time()
        upscaled = self.sr.upsample(image)
        end = time.time()
        print("[INFO] super resolution took {:.6f} seconds".format(
            end - start))
        # show the spatial dimensions of the super resolution image
        print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
                                           upscaled.shape[0]))

        cv2.imwrite("./data/pictures_out/SuperResolutionResult.jpg", image)