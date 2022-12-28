import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn import model_selection
import os, cv2, yaml, shutil, logging, albumentations

__author__ = 'Alex Klyuev'

TYPE_SIMPLE_RUN = True


class Configurations:
    """ Configurations Class for initialization project's parameters. """
    def __init__(self, dir_dataset, dir_yolo_dataset, stages, box_type='pascal_voc', aug_copies=15, test_size=0.3,
                 seed=42, verbose=1):
        """ Initialization Configurations object.

        :param dir_dataset: Path to original dataset of images.
        :param dir_yolo_dataset: Path to YOLO dataset what we need to compile.
        :param stages: Included pipeline elements ('create_labels' - create annotations/labels of the original images.
        'perform_augmentation' - perform augmentation to create new images and labels. 'compile_yolo_ds' - collect
        images and labels put it order to YOLO dataset to train it).
        :param box_type: Type of output coordinates for model. Possible types: 'pascal_voc' - normal values of the
        coordinates ([xmin, ymin, xmax, ymax]). 'yolo' - specified type of coordinates ([x, y, width, height]), where x,
        y - normalized values.
        :param aug_copies: Number of copies what will be created after augmentation.
        :param test_size: Ratio train/tess files for train.
        :param seed: Random state.
        :param verbose: Debug parameter. If verbose: 1 - weak debugging, 2 - strong debugging mode (show all logging).
        :return:
        """
        # Dataset paths:
        self.dir_dataset = dir_dataset
        self.dir_images = self.dir_dataset/'images'
        self.dir_labels = self.dir_dataset/'labels'
        self.dir_orig_images = self.dir_images/'orig'
        self.dir_orig_labels = self.dir_labels/'orig'
        self.dir_augmented_images = self.dir_images/'augmented'
        self.dir_augmented_labels = self.dir_labels/'augmented'
        # YOLO dataset paths:
        self.dir_yolo_dataset = dir_yolo_dataset
        self.dir_yolo_images = self.dir_yolo_dataset/'images'
        self.dir_yolo_labels = self.dir_yolo_dataset/'labels'
        self.dir_yolo_images_train = self.dir_yolo_images/'train'
        self.dir_yolo_images_test = self.dir_yolo_images/'test'
        self.dir_yolo_labels_train = self.dir_yolo_labels/'train'
        self.dir_yolo_labels_test = self.dir_yolo_labels/'test'
        self.yaml_fn = self.dir_yolo_dataset/f'{str(self.dir_yolo_dataset).split("/")[-1]}.yaml'
        # Formats:
        self.label_exts = ['txt']
        self.image_exts = ['png', 'jpg', 'gif', 'jpeg']
        # Creation dataset parameters:
        self.box_type = box_type
        self.copies = aug_copies
        self.test_size = test_size
        # Other configurations:
        self.stages = stages
        self.seed = seed
        self.verbose = verbose
        if self.verbose == 2: logging.info('Configurations initialized.')


class Utils:
    """ Class Utils with common simple function for operations with files. """
    def __init__(self, configs):
        """ Initialization Utils object.

        :param configs: Configuration attributes (check description of the Configurations class).
        :return:
        """
        # Dataset paths:
        self.dir_labels = configs.dir_labels
        self.dir_orig_images = configs.dir_orig_images
        self.dir_augmented_images = configs.dir_augmented_images
        self.dir_orig_labels = configs.dir_orig_labels
        self.dir_augmented_labels = configs.dir_augmented_labels
        # YOLO dataset paths:
        self.dir_yolo_dataset = configs.dir_yolo_dataset
        self.dir_yolo_images = configs.dir_yolo_images
        self.dir_yolo_labels = configs.dir_yolo_labels
        self.dir_yolo_images_train = configs.dir_yolo_images_train
        self.dir_yolo_images_test = configs.dir_yolo_images_test
        self.dir_yolo_labels_train = configs.dir_yolo_labels_train
        self.dir_yolo_labels_test = configs.dir_yolo_labels_test
        # Labels:
        self.labels = None
        # Others:
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('Utils initialized.')

    def init_files(self):
        """ Create directories for datasets.

        :return:
        """
        if self.verbose == 2: logging.info('Start files initialization.')
        paths = [
            self.dir_labels,
            self.dir_orig_labels,
            self.dir_augmented_images,
            self.dir_augmented_labels,
            self.dir_yolo_dataset,
            self.dir_yolo_images,
            self.dir_yolo_labels,
            self.dir_yolo_images_train,
            self.dir_yolo_images_test,
            self.dir_yolo_labels_train,
            self.dir_yolo_labels_test
        ]
        for p in paths:
            if not os.path.exists(str(p)):
                os.mkdir(str(p))
        if self.verbose == 2: logging.info('Files initialization finished.')

    @staticmethod
    def load_files_paths(d, ext):
        """ Load paths of files from directories and sub-directories (static method).

        :param d: Init directory where it starts to searching.
        :param ext: File extension.
        :return:
        """
        paths = list()
        for path, subdirs, files in os.walk(d):
            for name in files:
                if name.split('.')[-1] in ext:
                    paths.append(os.path.join(path, name))

        return paths

    def init_labels(self):
        """ Create labels based on directories what files is located.

        :return:
        """
        subfolders = [f.path for f in os.scandir(str(self.dir_orig_images)) if f.is_dir()]
        self.labels = {sf.split('/')[-1]: i for i, sf in enumerate(subfolders)}
        if self.verbose == 2:
            logging.info('Classes initialized:')
            for k in self.labels.keys():
                logging.info(f'{k} --> {self.labels[k]}')

    @staticmethod
    def save_label(data, label, dir_labels, sub_d, p):
        """ Save annotation for image to .txt file (label and coordinates, static method).

        :param data: Output coordinates.
        :param label: Image object label.
        :param dir_labels: Directory where annotations (labels) are located.
        :param sub_d: Directory for definitely class.
        :param p: Annotation (label) filename.
        :return:
        """
        path_to_save = dir_labels/sub_d/p
        row = f'{label} {round(data[0], 6)} {round(data[1], 6)} {round(data[2], 6)} {round(data[3], 6)}'
        if not os.path.exists('/'.join(str(path_to_save).split('/')[:-1])):
            os.mkdir('/'.join(str(path_to_save).split('/')[:-1]))
        with path_to_save.open('w') as writer:
            writer.write(row)

    @staticmethod
    def save_image(data, dir_images, sub_d, p):
        """ Save image .png file (static method).

        :param data: Image data.
        :param dir_images: Directory where images are located.
        :param sub_d: Directory for definitely class.
        :param p: Image filename.
        :return:
        """
        path_to_save = str(dir_images/sub_d/p)
        if not os.path.exists('/'.join(str(path_to_save).split('/')[:-1])):
            os.mkdir('/'.join(str(path_to_save).split('/')[:-1]))
        cv2.imwrite(path_to_save, data)

    @staticmethod
    def remove_files(dir_f):
        """ Remove all files from chosen directory (static method).

        :param dir_f: Chosen directory in what files must to be deleted.
        :return:
        """
        files = [f for f in os.listdir(dir_f) if os.path.isfile(os.path.join(dir_f, f))]
        for f in files:
            os.remove(f)


class AutoBoxing:
    """ Class AutoBoxing for auto creation bounding box coordinates using Canny filter for train YOLO. """
    def __init__(self, configs):
        """ Initialization AutoBoxing object.

        :param configs: Configuration attributes (check description of the Configurations class).
        :return:
        """
        self.box_type = configs.box_type
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('AutoBoxing initialized.')

    @staticmethod
    def detect_edges(filename):
        """ Detect edges of image for creation bounding box (static method).

        :param filename: Path to image file.
        :return: Extracted detected by filter coordinates of the object edges.
        """
        # Read image:
        img = cv2.imread(filename)
        # Convert to grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection:
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        return edges

    def convert_coordinates(self, box_coordinates, size):
        """ Convert normal coordinates to other types.

        :param box_coordinates: Input normal values of the object edges.
        :param size: Initial image size.
        :return: Transformed coordinates of the bounding box for chosen type.
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
        """ Create bounding box.

        :param filename: Path to image file.
        :return: Return bounding box coordinates.
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
    """ Class DataAugmentation for generating using random transformation for increasing dataset as reason. """
    def __init__(self, configs):
        """ Initialization DataAugmentation object.

        :param configs: Configuration attributes (check description of the Configurations class).
        :return:
        """
        # Transformation pipeline for augmentation:
        self.transform = albumentations.Compose([
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.RandomShadow(p=0.2),
            albumentations.RandomSnow(p=0.2),
            albumentations.RandomFog(),
            albumentations.BBoxSafeRandomCrop(p=0.35),
            albumentations.ShiftScaleRotate(),
            ], bbox_params=albumentations.BboxParams(format=configs.box_type)
        )
        # Other:
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('DataAugmentation initialized.')

    def perform_augmentation(self, image, bboxes):
        """ Perform augmentation for input image data and bounding box.

        :param image: Input image data array.
        :param bboxes: Input bounding box structure for transformation [coordinates, label].
        :return: Tuple of the transformed image array and bounding box.
        """
        transformed = self.transform(image=image, bboxes=bboxes)
        if self.verbose == 2:
            logging.info(f'Transformed image: {transformed["image"]}')
            logging.info(f'Transformed bounding box: {transformed["bboxes"]}')
        return transformed['image'], transformed['bboxes']


class DatasetCreator:
    """ Class DatasetCreator for a whole process from creation annotations for images to compiling dataset for train YOLO
    with augmented images and metadata for dataset. """
    def __init__(self, configs):
        """ Initialization DatasetCreator object.

        :param configs: Configuration attributes (check description of the Configurations class).
        :return:
        """
        # YOLO dataset paths:
        self.dir_yolo_dataset = configs.dir_yolo_dataset
        self.yaml_fn = configs.yaml_fn
        # Init Utils object:
        self.utils = Utils(configs=configs)
        self.utils.init_files()
        self.utils.init_labels()
        # Init AutoBoxing object:
        self.auto_boxing = AutoBoxing(configs=configs)
        # Init DataAugmentation object:
        self.data_augmentation = DataAugmentation(configs=configs)
        # Preparation dataset paraneters:
        self.copies = configs.copies
        self.test_size = configs.test_size
        # Formats:
        self.label_exts = configs.label_exts
        self.image_exts = configs.image_exts
        # Other:
        self.stages = configs.stages
        self.seed = configs.seed
        self.verbose = configs.verbose
        if self.verbose == 2: logging.info('DatasetCreator initialized.')

    def create_labels(self, dir_images, dir_labels):
        """ Create annotation files for initial images.

        :param dir_images: Path to directory where images are located.
        :param dir_labels: Path to directory where labels will be saved.
        :return:
        """
        if self.verbose: logging.info('Start to create labels.')
        images_paths = self.utils.load_files_paths(d=dir_images, ext=self.image_exts)
        for p in tqdm(images_paths, desc='Saving bbox coordinates:', disable=not self.verbose):
            bbox_c = self.auto_boxing.create_box(filename=p)
            label = self.utils.labels[p.split('/')[-2]]
            sub_d = p.split('/')[-2]
            label_fn = f'{p.split("/")[-1].split(".")[0]}.txt'
            self.utils.save_label(data=bbox_c, label=label, dir_labels=dir_labels, sub_d=sub_d, p=label_fn)

    def create_augmented_set(self, dir_images, dir_labels, dir_images_aug, dir_labels_aug):
        """ Create dataset of augmented images from original images.

        :param dir_images: Directory of the initial images.
        :param dir_labels: Directory of the initial labels.
        :param dir_images_aug: Directory where to save augmented images.
        :param dir_labels_aug: Directory where to save augmented labels.
        :return:
        """
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

    def train_test_split(self, dir_images_orig, dir_labels_orig, dir_images_aug, dir_labels_aug):
        """ Split paths on train and test groups for train YOLO.

        :param dir_images_orig: Directory of the initial images.
        :param dir_labels_orig: Directory of the initial labels.
        :param dir_images_aug: Directory where's saved augmented images.
        :param dir_labels_aug: Directory where's saved augmented labels.
        :return: List of separated on train and test paths to images/labels.
        """
        orig_images_p = self.utils.load_files_paths(d=dir_images_orig, ext=self.image_exts)
        orig_labels_p = self.utils.load_files_paths(d=dir_labels_orig, ext=self.label_exts)
        aug_images_p = self.utils.load_files_paths(d=dir_images_aug, ext=self.image_exts)
        aug_labels_p = self.utils.load_files_paths(d=dir_labels_aug, ext=self.label_exts)
        images_p = sorted(orig_images_p + aug_images_p)
        labels_p = sorted(orig_labels_p + aug_labels_p)

        return model_selection.train_test_split(
            images_p,
            labels_p,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True
        )

    def compile_ds_for_yolo(self, dir_images_orig, dir_labels_orig, dir_images_aug, dir_labels_aug, dir_yolo_train_images,
                            dir_yolo_train_labels, dir_yolo_test_images, dir_yolo_test_labels):
        """ Create data from original images performing autoboxing, augmentation and splitting dataset and sort it
        according needed architecture for train YOLO model for object detection task.

        :param dir_images_orig: Directory of the initial images.
        :param dir_labels_orig: Directory of the initial labels.
        :param dir_images_aug: Directory where to save augmented images.
        :param dir_labels_aug: Directory where to save augmented labels.
        :param dir_yolo_train_images: Directory of train images in YOLO dataset.
        :param dir_yolo_train_labels: Directory of train labels in YOLO dataset.
        :param dir_yolo_test_images: Directory of test images in YOLO dataset.
        :param dir_yolo_test_labels: Directory of test labels in YOLO dataset.
        :return:
        """
        paths = self.train_test_split(
            dir_images_orig=dir_images_orig,
            dir_labels_orig=dir_labels_orig,
            dir_images_aug=dir_images_aug,
            dir_labels_aug=dir_labels_aug
        )
        for p in tqdm(zip(paths[0], paths[2]), desc='Coping train files to YOLO ds', disable=not self.verbose):
            shutil.copy(src=p[0], dst=f'{dir_yolo_train_images}/{p[0].split("/")[-1]}')
            shutil.copy(src=p[1], dst=f'{dir_yolo_train_labels}/{p[1].split("/")[-1]}')
        for p in tqdm(zip(paths[1], paths[3]), desc='Coping test files to YOLO ds', disable=not self.verbose):
            shutil.copy(src=p[0], dst=f'{dir_yolo_test_images}/{p[0].split("/")[-1]}')
            shutil.copy(src=p[1], dst=f'{dir_yolo_test_labels}/{p[1].split("/")[-1]}')

        yaml_labels = {self.utils.labels[k]: k for k in self.utils.labels.keys()}
        yaml_content = {
            'path': f'{self.dir_yolo_dataset}',
            'train': 'images/train',
            'val': 'images/test',
            'test': 'images/test',
            'names': yaml_labels
        }
        with self.yaml_fn.open('w') as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

    def perform_dataset_creator(self, dir_images, dir_labels, dir_images_orig, dir_labels_orig, dir_images_aug,
                                dir_labels_aug, dir_yolo_train_images, dir_yolo_train_labels, dir_yolo_test_images,
                                dir_yolo_test_labels):
        """ Execute transformation pipeline for creating dataset for train YOLO.

        :param dir_images: Directory with all images.
        :param dir_labels: Directory with all labels.
        :param dir_images_orig: Directory of the initial images.
        :param dir_labels_orig: Directory of the labels for initial images.
        :param dir_images_aug: Directory where to save augmented images.
        :param dir_labels_aug: Directory where to save augmented labels.
        :param dir_yolo_train_images: Directory of train images in YOLO dataset.
        :param dir_yolo_train_labels: Directory of train labels in YOLO dataset.
        :param dir_yolo_test_images: Directory of test images in YOLO dataset.
        :param dir_yolo_test_labels: Directory of test labels in YOLO dataset.
        :return:
        """
        if 'create_labels' in self.stages:
            self.create_labels(dir_images=dir_images, dir_labels=dir_labels_orig)
        if 'perform_augmentation' in self.stages:
            self.create_augmented_set(
                dir_images=dir_images,
                dir_labels=dir_labels,
                dir_images_aug=dir_images_aug,
                dir_labels_aug=dir_labels_aug
            )
        if 'compile_yolo_ds' in self.stages:
            self.compile_ds_for_yolo(
                dir_images_orig=dir_images_orig,
                dir_labels_orig=dir_labels_orig,
                dir_images_aug=dir_images_aug,
                dir_labels_aug=dir_labels_aug,
                dir_yolo_train_images=dir_yolo_train_images,
                dir_yolo_train_labels=dir_yolo_train_labels,
                dir_yolo_test_images=dir_yolo_test_images,
                dir_yolo_test_labels=dir_yolo_test_labels
            )


if __name__ == '__main__':
    if TYPE_SIMPLE_RUN:
        # Init paths to datasets:
        current_dir = Path(__file__).parent
        dir_dataset = current_dir/'objects_dataset'
        dir_yolo_dataset = current_dir/'yolo_dataset'

        # Init configs:
        configs = Configurations(
            dir_dataset=dir_dataset,
            dir_yolo_dataset=dir_yolo_dataset,
            stages=['create_labels', 'perform_augmentation', 'compile_yolo_ds'],
            box_type='yolo',
            aug_copies=20,
            test_size=0.3,
            seed=42,
            verbose=1
        )
        # Init dataset_creator:
        ds_creator = DatasetCreator(configs=configs)
        # Perform pipeline execution:
        ds_creator.perform_dataset_creator(
            dir_images=configs.dir_images,
            dir_labels=configs.dir_labels,
            dir_images_orig=configs.dir_orig_images,
            dir_labels_orig=configs.dir_orig_labels,
            dir_images_aug=configs.dir_augmented_images,
            dir_labels_aug=configs.dir_augmented_labels,
            dir_yolo_train_images=configs.dir_yolo_images_train,
            dir_yolo_train_labels=configs.dir_yolo_labels_train,
            dir_yolo_test_images=configs.dir_yolo_images_test,
            dir_yolo_test_labels=configs.dir_yolo_labels_test
        )
    else:
        logging.info('Set global var "TYPE_SIMPLE_RUN" to True')
