import cv2
import numpy as np


class Configurations:
    def __init__(self):
        pass


class Utils:
    def __init__(self):
        pass


class AutoBoxing:
    def __init__(self, box_type='original'):
        self.box_type = box_type

    @staticmethod
    def detect_edges(filename):
        """

        :param filename:
        :return:
        """
        # Read image:
        img = cv2.imread(filename)
        # Convert to grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection:
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # Sobel Edge Detection:
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)    # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)    # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)   # Combined X and Y Sobel Edge Detection
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)   # Canny Edge Detection

        return edges

    def convert_coordinates(self, box_coordinates, size):
        """

        :param box_coordinates:
        :param size:
        :return:
        """
        if self.box_type == 'original':
            return (box_coordinates[0], box_coordinates[2]), (box_coordinates[1], box_coordinates[3])
        elif self.box_type == 'yolo':
            dw = 1. / size[0]
            dh = 1. / size[1]
            x = (box_coordinates[0] + box_coordinates[1]) / 2.0
            y = (box_coordinates[2] + box_coordinates[3]) / 2.0
            w = box_coordinates[1] - box_coordinates[0]
            h = box_coordinates[3] - box_coordinates[2]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh

            return x, y, w, h

    def create_box(self, edges):
        """

        :param edges:
        :return:
        """
        nonzero_x_coords = list()
        nonzero_y_coords = list()
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y][x] != 0:
                    nonzero_x_coords.append(x)
                    nonzero_y_coords.append(y)
        box_coordinates = (
            np.min(nonzero_x_coords),
            np.max(nonzero_x_coords),
            np.min(nonzero_y_coords),
            np.max(nonzero_y_coords)
        )
        box_coordinates = self.convert_coordinates(
            box_coordinates=box_coordinates,
            size=(edges.shape[0], edges.shape[1])
        )

        return box_coordinates



if __name__ == '__main__':
    ab = AutoBoxing(box_type='yolo')
    e = ab.detect_edges(filename='/home/aklyuev/PycharmProjects/ToyBoyDynamics/TOYBOY DYNAMICS/datasets/triangle_part/images/0.png')
    img = cv2.imread('/home/aklyuev/PycharmProjects/ToyBoyDynamics/TOYBOY DYNAMICS/datasets/triangle_part/images/0.png')
    box_c = ab.create_box(edges=e)
    #image = cv2.rectangle(img, box_c[0], box_c[1], (0, 0, 255), 2)
    #window_name = 'Image'
    #cv2.imshow(window_name, image)
    #cv2.waitKey(0)
