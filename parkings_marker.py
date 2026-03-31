import cv2
import json
import argparse
import numpy as np
import supervision as sv

class PolygonEditor:
    def __init__(self, video_source: str, output_json: str):
        self.video_source = video_source
        self.output_json = output_json

        self.polygons = []
        self.current_polygon = []
        self.frame = None
        self.display_frame = None

        # Load first frame
        self.load_first_frame()

        # Mouse callback
        cv2.namedWindow("Polygon Editor")
        cv2.setMouseCallback("Polygon Editor", self.mouse_callback)

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_source)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Could not read video source")

        self.frame = frame
        self.display_frame = frame.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))

    def draw(self):
        self.display_frame = self.frame.copy()

        # Draw existing polygons
        for polygon in self.polygons:
            pts = np.array(polygon, np.int32)
            cv2.polylines(self.display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw current polygon
        if len(self.current_polygon) > 0:
            pts = np.array(self.current_polygon, np.int32)
            cv2.polylines(self.display_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

            for point in self.current_polygon:
                cv2.circle(self.display_frame, point, 5, (0, 0, 255), -1)

    def save_polygons(self):
        data = {
            "video_source": self.video_source,
            "frame_resolution": [self.frame.shape[1], self.frame.shape[0]],
            "polygons": self.polygons
        }

        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved polygons to {self.output_json}")

    def run(self):
        print("Instructions:")
        print("Left click = add point")
        print("u = undo last point")
        print("n = save current polygon and start new one")
        print("s = save polygons to JSON")
        print("q = quit")

        while True:
            self.draw()
            cv2.imshow("Polygon Editor", self.display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("u"):  # Undo
                if self.current_polygon:
                    removed = self.current_polygon.pop()
                    print(f"Removed point: {removed}")

            elif key == ord("n"):  # Next polygon
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                    print(f"Polygon saved. Total polygons: {len(self.polygons)}")
                    self.current_polygon = []
                else:
                    print("Polygon needs at least 3 points")

            elif key == ord("s"):  # Save to JSON
                self.save_polygons()

            elif key == ord("q"):  # Quit
                break

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Polygon Editor")
    parser.add_argument("--video", type=str, required=True, help="Path to video or stream")
    parser.add_argument("--output", type=str, default="output/polygons.json", help="Output JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    editor = PolygonEditor(args.video, args.output)
    editor.run()