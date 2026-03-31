import cv2
import logging
from typing import List, Any
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from rfdetr import RFDETRNano
import supervision as sv
from utils import get_device, parse_video_source
import argparse

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

VIDEO_PATH = "example_media/parking_2.mp4"
OUTPUT_VIDEO_PATH = "output/saved_video.mp4"
MODEL_PATH = "models/VisDrone_model.pth"
PREDICT_THRESHOLD = 0.1

class DetectionApp:
    def __init__(self, video_source: str, show: bool, save: bool, output_path: str, track: bool):
        self.show = show
        self.save = save
        self.track = track
        self.output_path = output_path
        self.last_frame = None
        self.video_source = parse_video_source(video_source)

        # Model
        self.model = RFDETRNano(pretrain_weights=MODEL_PATH, device=get_device())
        self.model.optimize_for_inference()

        # Supervision tools
        self.tracker = sv.ByteTrack()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

        # Video info - used for saving
        if self.save:
            self.video_info = sv.VideoInfo.from_video_path(self.video_source)
        
        # State
        self.paused = False

        # Pipeline
        self.pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=self.video_source,
            on_video_frame=self.infer,
            on_prediction=self.on_prediction,
        )

    def process_predicted_frame(self, prediction, video_frame):
        # Project logic comes here
        pass




##### Detection and tracking functions #####


    def on_prediction(self, prediction, video_frame):
        if self.track:
            prediction = self.track_objects(prediction, video_frame)

        if len(prediction) > 0:
            car_mask = prediction.class_id == 3
            prediction = prediction[car_mask]

        self.process_predicted_frame(prediction=prediction, video_frame=video_frame)

        if self.show or self.save:
            self.annotate_image(prediction, video_frame)
            if self.show:
                self.visualization()
            if self.save:
                self.save_video()

    def infer(self, video_frames: List[VideoFrame]) -> List[Any]:
        predictions = self.model.predict(
            [v.image for v in video_frames],
            threshold=PREDICT_THRESHOLD
            )
        return [predictions]

    def track_objects(self, prediction, video_frame):
        tracked_detections = self.tracker.update_with_detections(prediction)
        return tracked_detections

    def annotate_image(self, prediction, video_frame):
        if self.track and prediction.tracker_id is not None:
            labels = [
                f"Car {conf:.2f} ID:{int(track_id)}"
                for track_id, conf in zip(
                    prediction.tracker_id,
                    prediction.confidence
                )
            ]
        else:
            labels = [
                f"Car {conf:.2f}"
                for conf in prediction.confidence
            ]

        annotated_image = video_frame.image.copy()

        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=prediction
        )

        annotated_image = self.label_annotator.annotate(
            annotated_image,
            detections=prediction,
            labels=labels
        )

        self.last_frame = annotated_image

    def save_video(self):
        if self.last_frame is not None:
            self.sink.write_frame(self.last_frame)

    def visualization(self):
        cv2.imshow("Predictions", self.last_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.pipeline.terminate()
            exit()

        elif key == ord(" "):
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def run(self):
        if self.save:
            with sv.VideoSink(self.output_path, self.video_info) as sink:
                self.sink = sink
                self.pipeline.start()
                self.pipeline.join()
        else:
            self.pipeline.start()
            self.pipeline.join()

        cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(description="Detection App")

    parser.add_argument("--video", type=str, default=VIDEO_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH)
    
    parser.add_argument("--track", action="store_true", help="Enable tracking")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = DetectionApp(
        video_source=args.video,
        show=args.show,
        save=args.save,
        track=args.track,
        output_path=args.output
    )
    app.run()