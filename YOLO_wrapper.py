import cv2, torch


class YOLOWrapper:
    def __init__(self):
        self.yolo_model = torch.hub.load('/home/aklyuev/PycharmProjects/yolov5', 'custom', path='/home/aklyuev/PycharmProjects/yolov5/runs/train/exp32/weights/best.pt', source='local')

    def perform_detection(self, image):
        return self.yolo_model(image).xyxy[0].numpy()

    def apply_yolo(self, image):
        prediction = self.perform_detection(image=image)
        bb_coords, bb_confidence, bb_labels = list(), list(), list()
        for i in range(prediction.shape[0]):
            bb_coords.append([int(prediction[i][0]), int(prediction[i][1]), int(prediction[i][2]), int(prediction[i][3])])
            bb_confidence.append(prediction[i][4])
            bb_labels.append(prediction[i][5])

        return bb_coords, bb_confidence, bb_labels


if __name__ == '__main__':
    ym = YOLOWrapper()
    video_stream = cv2.VideoCapture(0)
    is_stream_stop = False
    while not is_stream_stop:
        # Capture frame-by-frame
        ret, frame = video_stream.read()
        coords, conf, labels = ym.apply_yolo(image=frame)
        if len(coords) > 0:
            for box in coords:
                print(box)
                box_image = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        else:
            box_image = frame
        cv2.imshow('Frame', box_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_stream_stop = True
    video_stream.release()
    # Closes all the frames
    cv2.destroyAllWindows()
