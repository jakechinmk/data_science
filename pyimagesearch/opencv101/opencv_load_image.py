import argparse
import cv2
from matplotlib import pyplot as plt

def plt_imshow(title, image):
    """
    convert the image frame bgr to rgb color space and display it
    :param title: title for the window frame
    :param image: image to show
    :return: noe
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='path to input image')
    ap.add_argument('-o', '--output', required=True,
                    help='path for saving image')
    ap.add_argument('-s', '--save', required=True,
                    help='save image?')
    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])
    (h, w, c) = image.shape[:3]

    print('width: {} pixels'.format(image.shape[1]))
    print('height: {}  pixels'.format(image.shape[0]))
    print('channels: {}'.format(image.shape[2]))

    cv2.imshow('Image', image)
    cv2.waitKey(0)

    if args['save']:
        cv2.imwrite(args['output'], image)