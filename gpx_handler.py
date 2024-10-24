import gpxpy
from datetime import timedelta
import os
import cv2
import json

def process_gpx(gpx_path, video_path, author, device, category, process_mode):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join("outputs", video_name)

    os.makedirs(output_folder, exist_ok=True)

    # Frames and JSON are saved inside the output folder
    frames_folder = os.path.join(output_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)

    # Load the GPX file
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation": point.elevation,
                    "time": point.time
                })

    if not points:
        raise ValueError("GPX file does not contain any points.")

    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_start_time = points[0]["time"]
    time_per_frame = timedelta(seconds=1 / frame_rate)
    frame_times = [video_start_time + i * time_per_frame for i in range(total_frames)]

    point_idx = 0
    point_count = len(points)
    extracted_frames = []
    last_saved_time = video_start_time

    for frame_number, frame_time in enumerate(frame_times):
        success, frame = video_capture.read()
        if not success:
            break

        if point_idx >= point_count:
            break

        while point_idx < point_count - 1 and points[point_idx + 1]["time"] <= frame_time:
            point_idx += 1

        point = points[point_idx]
        if point_idx < point_count - 1:
            next_point = points[point_idx + 1]
            time_diff = (next_point["time"] - point["time"]).total_seconds()
            if time_diff > 0:
                ratio = (frame_time - point["time"]).total_seconds() / time_diff
                lat = point["lat"] + ratio * (next_point["lat"] - point["lat"])
                lon = point["lon"] + ratio * (next_point["lon"] - point["lon"])
                elevation = point["elevation"] + ratio * (next_point["elevation"] - point["elevation"])
            else:
                lat, lon, elevation = point["lat"], point["lon"], point["elevation"]
        else:
            lat, lon, elevation = point["lat"], point["lon"], point["elevation"]

        if (frame_time - last_saved_time).total_seconds() >= 2:
            frame_filename = f"frame_{frame_number:04d}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

            extracted_frames.append({
                "frame_number": frame_number,
                "image_url": f"/map/{video_name}/frames/{frame_filename}",
                "latitude": lat,
                "longitude": lon,
                "timestamp": frame_time.isoformat() + 'Z',
                "elevation": elevation,
                "author": author,
                "device": device,
                "category": category,
                "process": process_mode
            })

            last_saved_time = frame_time

    json_output_path = os.path.join(output_folder, f"{video_name}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(extracted_frames, json_file, indent=4)

    video_capture.release()
