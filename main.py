import os
import tempfile
from collections import defaultdict, deque

import cv2 as cv
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO


st.set_page_config(
    page_title="Vehicle Speed Detection",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO("yolo11m.pt")


model = load_model()


class Cam2WorldMapper:

    def __init__(self):
        self.M = None

    def find_perspective_transform(
        self,
        image_pts,
        world_pts
    ):

        image_pts = np.asarray(
            image_pts,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        world_pts = np.asarray(
            world_pts,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        self.M = cv.getPerspectiveTransform(
            image_pts,
            world_pts
        )

    def map(self, points):

        points = np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        return cv.perspectiveTransform(
            points,
            self.M
        ).reshape(-1, 2)


class SpeedEstimator:

    def __init__(
        self,
        mapper,
        fps
    ):

        self.mapper = mapper
        self.fps = fps

        self.tracks = defaultdict(
            lambda: deque(
                maxlen=45
            )
        )

        self.speeds = defaultdict(
            lambda: deque(
                maxlen=10
            )
        )

        self.current_speed = {}

    def update(
        self,
        track_id,
        image_point,
        frame_number
    ):

        world_point = self.mapper.map(
            [image_point]
        )[0]

        self.tracks[
            track_id
        ].append(
            (
                frame_number,
                world_point
            )
        )

        history = self.tracks[
            track_id
        ]

        if len(history) < 15:

            return None

        old_frame, old_world = (
            history[0]
        )

        new_frame, new_world = (
            history[-1]
        )

        frame_delta = (
            new_frame -
            old_frame
        )

        if frame_delta <= 0:

            return None

        elapsed = (
            frame_delta /
            self.fps
        )

        dx = (
            new_world[0] -
            old_world[0]
        )

        dy = (
            new_world[1] -
            old_world[1]
        )

        distance = np.sqrt(
            dx * dx +
            dy * dy
        )

        speed_ms = (
            distance /
            elapsed
        )

        speed_kmh = (
            speed_ms * 3.6
        )

        speed_kmh = max(
            0,
            speed_kmh - 50
        )

        if speed_kmh < 3:

            return self.current_speed.get(
                track_id,
                None
            )

        if speed_kmh > 120:

            return self.current_speed.get(
                track_id,
                None
            )

        self.speeds[
            track_id
        ].append(
            speed_kmh
        )

        speed = np.median(
            self.speeds[
                track_id
            ]
        )

        self.current_speed[
            track_id
        ] = int(speed)

        return int(speed)


def process_video(
    input_video
):

    cap = cv.VideoCapture(
        input_video
    )

    fps = cap.get(
        cv.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        fps = 30

    image_pts = [

        (
            800 / 1920 * width,
            410 / 900 * height
        ),

        (
            1125 / 1920 * width,
            410 / 900 * height
        ),

        (
            1920 / 1920 * width,
            850 / 900 * height
        ),

        (
            0 / 1920 * width,
            850 / 900 * height
        )
    ]

    world_pts = [

        (0, 0),

        (32, 0),

        (32, 140),

        (0, 140)
    ]

    mapper = Cam2WorldMapper()

    mapper.find_perspective_transform(
        image_pts,
        world_pts
    )

    speedometer = SpeedEstimator(
        mapper,
        fps
    )

    zone_polygon = np.array([

        (
            0,
            int(
                410 / 900 * height
            )
        ),

        (
            width,
            int(
                410 / 900 * height
            )
        ),

        (
            width,
            height
        ),

        (
            0,
            height
        )
    ])

    zone = sv.PolygonZone(
        zone_polygon,
        (
            sv.Position.TOP_CENTER,
            sv.Position.BOTTOM_CENTER
        )
    )

    colors = (

        "#007fff",
        "#0072e6",
        "#0066cc",
        "#0059b3",
        "#004c99",
        "#004080",
        "#003366",
        "#00264d"

    )

    color_palette = sv.ColorPalette(
        list(
            map(
                sv.Color.from_hex,
                colors
            )
        )
    )

    bbox_annotator = sv.BoxAnnotator(
        color=color_palette,
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK
    )

    trace_annotator = sv.TraceAnnotator(
        color=color_palette,
        position=sv.Position.CENTER,
        thickness=2,
        trace_length=int(fps),
        color_lookup=sv.ColorLookup.TRACK
    )

    label_annotator = sv.RichLabelAnnotator(
        color=color_palette,
        border_radius=2,
        font_size=16,
        color_lookup=sv.ColorLookup.TRACK,
        text_padding=6
    )

    st.subheader(
        "🚗 Live Vehicle Speed Detection"
    )

    video_window = st.empty()

    progress_bar = st.progress(0)

    status = st.empty()

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame_number += 1

        results = model.track(

            frame,

            classes=[
                2,
                5,
                7
            ],

            conf=0.45,

            imgsz=640,

            persist=True,

            verbose=False,

            tracker="bytetrack.yaml"
        )

        detection = (
            sv.Detections.from_ultralytics(
                results[0]
            )
        )

        if len(detection) > 0:

            detection = detection[
                zone.trigger(
                    detections=detection
                )
            ]

        labels = []

        if (

            len(detection) > 0

            and

            detection.tracker_id
            is not None

        ):

            for i, track_id in enumerate(
                detection.tracker_id
            ):

                track_id = int(
                    track_id
                )

                x1, y1, x2, y2 = (
                    detection.xyxy[i]
                )

                bottom_center = (

                    float(
                        (x1 + x2) / 2
                    ),

                    float(y2)

                )

                speed = speedometer.update(

                    track_id,

                    bottom_center,

                    frame_number

                )

                if speed is None:

                    labels.append(
                        f"ID {track_id} | -- km/h"
                    )

                else:

                    labels.append(
                        f"ID {track_id} | "
                        f"{speed} km/h"
                    )

        if len(detection) > 0:

            frame = bbox_annotator.annotate(
                frame,
                detection
            )

            frame = trace_annotator.annotate(
                frame,
                detection
            )

            frame = label_annotator.annotate(
                frame,
                detection,
                labels=labels
            )

        frame_rgb = cv.cvtColor(
            frame,
            cv.COLOR_BGR2RGB
        )

        video_window.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        if total_frames > 0:

            progress_bar.progress(

                min(

                    frame_number /
                    total_frames,

                    1.0

                )

            )

        status.text(

            f"Processing "
            f"{frame_number}/"
            f"{total_frames}"

        )

    cap.release()

    progress_bar.empty()

    status.success(
        "Processing completed!"
    )

    video_window.empty()


st.title(
    "🚘 Vehicle Speed Detection"
)

st.write(
    "Upload a traffic video to start detection."
)

uploaded_file = st.file_uploader(

    "Upload Video",

    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]

)

if uploaded_file is not None:

    temp_dir = tempfile.mkdtemp()

    input_path = os.path.join(
        temp_dir,
        "input.mp4"
    )

    with open(
        input_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    process_video(
        input_path
    )