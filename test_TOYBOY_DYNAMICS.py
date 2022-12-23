from pathlib import Path
import cv2, sys, logging, unittest
from utils import Configurations, DatasetCreator, AutoBoxing

format = '%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format, force=True)
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format, force=True)
logging.info('logger on')

__author__ = 'Alex Klyuev'

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
    aug_copies=15,
    test_size=0.3,
    seed=42,
    verbose=1
)
# Init dataset_creator:
ds_creator = DatasetCreator(configs=configs)
# Init AutoBoxing:
auto_boxing = AutoBoxing(configs=configs)


class test_TOYBOY_DYNAMICS(unittest.TestCase):
    # Check how AutoBoxing works with textured images (mono-colored background):
    def test_edges_for_textured_image(self):
        auto_boxing.box_type = 'pascal_voc'
        bbox_coords = auto_boxing.create_box(filename=f'{current_dir}/test/textured_object.jpg')
        textured_image = cv2.imread(f'{current_dir}/test/textured_object.jpg')
        box_image = cv2.rectangle(
            textured_image,
            (bbox_coords[0], bbox_coords[1]),
            (bbox_coords[2], bbox_coords[3]),
            color=(0, 0, 255),
            thickness=2
        )
        cv2.imshow('Test', box_image)
        cv2.waitKey(0)


