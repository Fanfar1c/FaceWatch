from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import time
from collections import defaultdict, deque


CONFIDENCE_THRESHOLD = 0.3
TARGET_VIDEO_PATH = "vehicles-result.mp4"

model = YOLO("yolov8x.pt")
frame_generator = sv.get_video_frames_generator('VID_20240330_164210.mp4')
bounding_box_annotator = sv.BoundingBoxAnnotator()
classes2 = [2,5]
tracker = sv.ByteTrack()
video_info = sv.VideoInfo.from_video_path(video_path='VID_20240330_164210.mp4')
label_annotator = sv.LabelAnnotator()



SOURCE = np.array([
    [918, 390], 
    [1011, 390], 
    [1239, 1012],
    [503, 1000] 

])

TARGET = np.array([
    [0, 0],
    [10, 0],
    [10, 120],
    [0, 120],
])


polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh
)

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

thickness = sv.calculate_dynamic_line_thickness(
    resolution_wh=video_info.resolution_wh
)

text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=video_info.resolution_wh
)

trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)


label_annotator = sv.LabelAnnotator(
    text_scale=0.8,
    text_thickness=thickness,
    text_position=sv.Position.TOP_LEFT
)

blue_color = np.array([0, 0, 255])


with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    
    for frame in frame_generator:
        result = model(frame,classes = classes2)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]
        
        detections = detections[polygon_zone.trigger(detections)]

        detections = tracker.update_with_detections(detections)
        
        points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
        
        points = view_transformer.transform_points(points=points).astype(int)

            # положение обнаружения магазина
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
        
        labels = []
            
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")    

        
        #Ннаписать сверху label 
        # labels = [
        #     f"#{tracker_id} {result.names[class_id]}"
        #     for class_id, tracker_id
        #     in zip(detections.class_id, detections.tracker_id)
        # ]
        #Ннаписать сверху label 
        
        annotated_frame = frame.copy()
        
        #Линия поверх машин
        annotated_frame = trace_annotator.annotate(
                annotated_frame, detections=detections
            )
        #Линия поверх машин
        
        
        #Нарисовать четыреугольник на машине
        annotated_frame = bounding_box_annotator.annotate(
            annotated_frame, detections=detections)
        #Нарисовать четыреугольник на машине
        
        annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
        )
        
        #Ннаписать сверху label 
        # label_annotator.annotate(
        #     annotated_frame, detections=detections, labels=labels)
        #Ннаписать сверху label 
        # sink.write_frame(annotated_frame)
        
        frame2 = np.array(annotated_frame, dtype=np.uint8)
        
        cv2.imshow('Frame', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break

    cv2.destroyAllWindows() 
    



