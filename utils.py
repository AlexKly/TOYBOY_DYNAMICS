import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn import model_selection
import os, sys, cv2, logging, albumentations


format = '%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format, force=True)
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format, force=True)
logging.info('logger on')

__author__ = 'Alex Klyuev'


class Configurations:
    def __init__(self, dir_dataset, box_type='pascal_voc', aug_copies=15, test_size=0.3, seed=42, verbose=1):
        # Dataset paths:
        self.dir_dataset = dir_dataset
        self.dir_images = self.dir_dataset/'images'
        self.dir_labels = self.dir_dataset/'labels'
        self.dir_orig_images = self.dir_images/'orig'
        self.dir_orig_labels = self.dir_labels/'orig'
        self.dir_augmented_images = self.dir_images/'augmented'
        self.dir_augmented_labels = self.dir_labels/'augmented'
        # Formats:
        self.label_exts = ['txt']
        self.image_exts = ['png', 'jpg', 'gif', 'jpeg']
        # Other:
        self.box_type = box_type
        self.copies = aug_copies
        self.test_size = test_size
        self.seed = seed
        self.verbose = verbose
        if self.verbose == 2: logging.info('Configurations initialized.')


class Utils:
    def __init__(self, configs):
        self.dir_labels = configs.dir_labels
        self.dir_orig_images = configs.dir_orig_images
        self.dir_augmented_images = configs.dir_augmented_images
        self.dir_orig_labels = configs.dir_orig_labels
        self.dir_augmented_labels = configs.dir_augmented_labels
        self.labels = None
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('Utils initialized.')

    def init_files(self):
        if self.verbose == 2: logging.info('Start files initialization.')
        paths = [self.dir_labels, self.dir_orig_labels, self.dir_augmented_images, self.dir_augmented_labels]
        for p in paths:
            if not os.path.exists(str(p)):
                os.mkdir(str(p))
        if self.verbose == 2: logging.info('Files initialization finished.')

    @staticmethod
    def load_files_paths(d, ext):
        paths = list()
        for path, subdirs, files in os.walk(d):
            for name in files:
                if name.split('.')[-1] in ext:
                    paths.append(os.path.join(path, name))

        return paths

    def init_labels(self):
        subfolders = [f.path for f in os.scandir(str(self.dir_orig_images)) if f.is_dir()]
        self.labels = {sf.split('/')[-1]: i for i, sf in enumerate(subfolders)}
        if self.verbose == 2:
            logging.info('Classes initialized:')
            for k in self.labels.keys():
                logging.info(f'{k} --> {self.labels[k]}')

    @staticmethod
    def save_label(data, label, dir_labels, sub_d, p):
        path_to_save = dir_labels/sub_d/p
        row = f'{label} {round(data[0], 6)} {round(data[1], 6)} {round(data[2], 6)} {round(data[3], 6)}'
        if not os.path.exists('/'.join(str(path_to_save).split('/')[:-1])):
            os.mkdir('/'.join(str(path_to_save).split('/')[:-1]))
        with path_to_save.open('w') as writer:
            writer.write(row)

    @staticmethod
    def save_image(data, dir_images, sub_d, p):
        path_to_save = str(dir_images/sub_d/p)
        if not os.path.exists('/'.join(str(path_to_save).split('/')[:-1])):
            os.mkdir('/'.join(str(path_to_save).split('/')[:-1]))
        cv2.imwrite(path_to_save, data)


class AutoBoxing:
    def __init__(self, configs):
        self.box_type = configs.box_type
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('AutoBoxing initialized.')

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
        if self.box_type == 'pascal_voc':
            return box_coordinates[0], box_coordinates[2], box_coordinates[1], box_coordinates[3]
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

    def create_box(self, filename):
        """

        :param filename:
        :return:
        """
        edges = self.detect_edges(filename=filename)
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


class DataAugmentation:
    def __init__(self, configs):
        self.transform = albumentations.Compose(
            [
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.2),
                albumentations.RandomShadow(p=0.2),
                albumentations.RandomSnow(p=0.2),
                albumentations.RandomFog(),
            ],
            bbox_params=albumentations.BboxParams(format=configs.box_type)
        )
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('DataAugmentation initialized.')

    def perform_augmentation(self, image, bboxes):
        transformed = self.transform(image=image, bboxes=bboxes)
        if self.verbose == 2:
            logging.info(f'Transformed image: {transformed["image"]}')
            logging.info(f'Transformed bounding box: {transformed["bboxes"]}')
        return transformed['image'], transformed['bboxes']


class DatasetCreator:
    def __init__(self, configs):
        self.utils = Utils(configs=configs)
        self.utils.init_files()
        self.utils.init_labels()
        self.auto_boxing = AutoBoxing(configs=configs)
        self.data_augmentation = DataAugmentation(configs=configs)
        self.copies = configs.copies
        self.test_size = configs.test_size
        self.label_exts = configs.label_exts
        self.image_exts = configs.image_exts
        self.seed = configs.seed
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('DatasetCreator initialized.')

    def create_labels(self, dir_images, dir_labels):
        if self.verbose: logging.info('Start to create labels.')
        images_paths = self.utils.load_files_paths(d=dir_images, ext=self.image_exts)
        for p in tqdm(images_paths, desc='Saving bbox coordinates:', disable=not self.verbose):
            bbox_c = self.auto_boxing.create_box(filename=p)
            label = self.utils.labels[p.split('/')[-2]]
            sub_d = p.split('/')[-2]
            label_fn = f'{p.split("/")[-1].split(".")[0]}.txt'
            self.utils.save_label(data=bbox_c, label=label, dir_labels=dir_labels, sub_d=sub_d, p=label_fn)

    def create_augmented_set(self, dir_images, dir_labels, dir_images_aug, dir_labels_aug):
        if self.verbose: logging.info('Start to create augmented set.')
        images_paths = sorted(self.utils.load_files_paths(d=dir_images, ext=self.image_exts))
        labels_paths = sorted(self.utils.load_files_paths(d=dir_labels, ext=self.label_exts))
        for paths in tqdm(zip(images_paths, labels_paths), desc='Performing augmentation:', disable=not self.verbose):
            image = cv2.imread(paths[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with open(paths[1], 'r') as reader: annotations = reader.readlines()[0].replace('\n', '')
            label = annotations.split(' ')[0]
            coordinates = [float(coor) for coor in annotations.split(' ')[1:]]
            bboxes = [coordinates + [label]]
            for i in range(self.copies):
                transformed_image, transformed_bbox = self.data_augmentation.perform_augmentation(image=image, bboxes=bboxes)
                sub_d = paths[0].split('/')[-2]
                label_fn_im = f'{paths[0].split("/")[-1].split(".")[0]}_{i}.png'
                label_fn_lab = f'{paths[0].split("/")[-1].split(".")[0]}_{i}.txt'
                self.utils.save_image(
                    data=transformed_image,
                    dir_images=dir_images_aug,
                    sub_d=sub_d,
                    p=label_fn_im
                )
                self.utils.save_label(
                    data=transformed_bbox[0][:-1],
                    label=transformed_bbox[0][-1],
                    dir_labels=dir_labels_aug,
                    sub_d=sub_d,
                    p=label_fn_lab
                )

    def train_test_split(self, dir_images, dir_labels, dir_images_aug, dir_labels_aug):
        orig_images_p = utils.load_files_paths(d=dir_images, ext=self.image_exts)
        orig_labels_p = utils.load_files_paths(d=dir_labels, ext=self.label_exts)
        aug_images_p = utils.load_files_paths(d=dir_images_aug, ext=self.image_exts)
        aug_labels_p = utils.load_files_paths(d=dir_labels_aug, ext=self.label_exts)
        images_p = sorted(orig_images_p + aug_images_p)
        labels_p = sorted(orig_labels_p + aug_labels_p)

        return model_selection.train_test_split(
            images_p,
            labels_p,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True
        )

    def compile_ds_for_yolo(self, dir_images, dir_labels, dir_images_aug, dir_labels_aug):
        paths = self.train_test_split(
            dir_images=dir_images,
            dir_labels=dir_labels,
            dir_images_aug=dir_images_aug,
            dir_labels_aug=dir_labels_aug
        )
        




if __name__ == '__main__':
    configs = Configurations(dir_dataset=Path('/home/aklyuev/PycharmProjects/TOYBOY_DYNAMICS/objects_dataset'), box_type='yolo')
    utils = Utils(configs=configs)
    dc = DatasetCreator(configs=configs)
    t = dc.train_test_split(
        dir_images=configs.dir_orig_images,
        dir_labels=configs.dir_orig_labels,
        dir_images_aug=configs.dir_augmented_images,
        dir_labels_aug=configs.dir_augmented_labels
    )

    #ab = AutoBoxing(box_type='yolo')
    #e = ab.detect_edges(filename='/home/aklyuev/PycharmProjects/ToyBoyDynamics/TOYBOY DYNAMICS/datasets/triangle_part/images/0.png')
    #img = cv2.imread('/home/aklyuev/PycharmProjects/ToyBoyDynamics/TOYBOY DYNAMICS/datasets/triangle_part/images/0.png')
    #box_c = ab.create_box(edges=e)
    #print(box_c)
    #image = cv2.rectangle(img, box_c[0], box_c[1], (0, 0, 255), 2)
    #window_name = 'Image'
    #cv2.imshow(window_name, image)
    #cv2.waitKey(0)
