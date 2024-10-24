import cv2
import numpy as np

# Function to pixelate the image based on the number of sectors
def pixelate_frame(frame, num_sectors, resize_factor):
    height, width, _ = frame.shape

    # Calculate the number of rows and columns for sectors
    num_rows = int(np.ceil(np.sqrt(num_sectors)))  # Rows
    num_cols = int(np.ceil(num_sectors / num_rows))  # Columns

    # Calculate sector dimensions
    sector_height = height / num_rows
    sector_width = width / num_cols

    # Create a pixelated version of the frame
    pixelated_frame = np.zeros_like(frame)

    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the region of interest
            start_y = int(i * sector_height)
            start_x = int(j * sector_width)
            end_y = int((i + 1) * sector_height)
            end_x = int((j + 1) * sector_width)

            # Ensure that the sector dimensions do not exceed the frame dimensions
            end_y = min(end_y, height)
            end_x = min(end_x, width)

            # Get the current sector
            sector = frame[start_y:end_y, start_x:end_x]

            # Calculate the mean color of the sector
            mean_color = np.mean(sector, axis=(0, 1)).astype(int)

            # Fill the pixelated frame with the mean color
            pixelated_frame[start_y:end_y, start_x:end_x] = mean_color

    return pixelated_frame

# Function to pixelate the video
def pixelate_video(video_path, output_video_path, num_sectors, resize_factor):
    cap = cv2.VideoCapture(video_path)
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the new frame dimensions based on resize factor
    frame_width = original_frame_width // resize_factor
    frame_height = original_frame_height // resize_factor

    # Create VideoWriter object with native resolution for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_frame_width, original_frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        # Pixelate the resized frame
        pixelated_frame = pixelate_frame(resized_frame, num_sectors, resize_factor)

        # Upscale the pixelated frame back to original size
        pixelated_frame_upscaled = cv2.resize(pixelated_frame, (original_frame_width, original_frame_height))
        out.write(pixelated_frame_upscaled)

    cap.release()
    out.release()
    print(f"Pixelated video saved as {output_video_path}")

