from PIL import Image
from DeepTools import CycleGAN, Recognize, SuperResolution


def function():
    C = CycleGAN()
    R = Recognize()
    S = SuperResolution()
    while(True):
        w = input('type in the function number：\n--------------------\n1.CycleGAN\n2.Recognize\n3.quit\n--------------------\n')
        if w =='1':
            path = input('type in the image\'s path：')
            C(path)
        elif w=='2':
            path = input('type in the image\'s path：')
            R(path)
        elif w=='3':
            path = input('type in the image\'s path：')
            S(path)
        elif w=='0':
            print('Bye!')
            break
        else:
            print('error occurs in your input！')

function()
