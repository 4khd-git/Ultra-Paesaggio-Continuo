import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter1d

# Function to process a single frame
def process_frame(frame, num_dominant_colors, resize_factor, smooth_factor):
    def rgb_to_hsv(rgb):
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    def get_overall_dominant_colors(pixels, num_colors):
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        return dominant_colors

    def calculate_color_percentages(frame, dominant_colors):
        pixels = frame.reshape((-1, 3))
        distances = np.sqrt(((pixels[:, np.newaxis] - dominant_colors) ** 2).sum(axis=2))
        closest_color_indices = np.argmin(distances, axis=1)
        unique, counts = np.unique(closest_color_indices, return_counts=True)
        percentages = np.zeros(len(dominant_colors))
        percentages[unique] = counts / len(pixels)
        return percentages

    def create_color_bar_fixed_position(dominant_colors, percentages, frame_height, frame_width):
        dominant_colors_hsv = [rgb_to_hsv(color) for color in dominant_colors]
        dominant_colors_sorted = [color for _, color in
                                  sorted(zip(dominant_colors_hsv, dominant_colors), key=lambda x: x[0][0])]
        percentages_sorted = [x for _, x in sorted(zip(dominant_colors_hsv, percentages), key=lambda x: x[0][0])]

        color_bar = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        current_x = 0

        for i, color in enumerate(dominant_colors_sorted):
            if percentages_sorted[i] > 0:
                bar_width = int(percentages_sorted[i] * frame_width)
                next_x = current_x + bar_width

                next_color = dominant_colors_sorted[i + 1] if i < len(dominant_colors_sorted) - 1 else color

                # Smooth transition
                for x in range(current_x, next_x):
                    if next_x > current_x:  # Ensure there is a range to blend
                        blend_factor = (x - current_x) / (next_x - current_x)
                        blended_color = (1 - blend_factor) * color + blend_factor * next_color
                    else:
                        blended_color = color  # No blending if no range

                    color_bar[:, x] = blended_color

                current_x = next_x

        # Fill remaining width with last color if needed
        if current_x < frame_width:
            color_bar[:, current_x:] = dominant_colors_sorted[-1]

        return color_bar

    # Resize frame for faster processing
    frame_height, frame_width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (frame_width // resize_factor, frame_height // resize_factor))

    # Flatten pixels and find dominant colors
    pixels = resized_frame.reshape(-1, 3)
    dominant_colors = get_overall_dominant_colors(pixels, num_dominant_colors)

    # Calculate percentages and create a color bar
    percentages = calculate_color_percentages(resized_frame, dominant_colors)
    smoothed_percentages = uniform_filter1d(np.array([percentages]), size=smooth_factor, axis=0)[0]
    color_bar = create_color_bar_fixed_position(dominant_colors, smoothed_percentages, frame_height, frame_width)

    return color_bar


def process_video(video_path, output_video_path, num_dominant_colors, resize_factor, smooth_factor):
    def rgb_to_hsv(rgb):
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    def get_overall_dominant_colors(all_pixels, num_colors):
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(all_pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        return dominant_colors

    def calculate_color_percentages(frame, dominant_colors):
        pixels = frame.reshape((-1, 3))
        distances = np.sqrt(((pixels[:, np.newaxis] - dominant_colors) ** 2).sum(axis=2))
        closest_color_indices = np.argmin(distances, axis=1)
        unique, counts = np.unique(closest_color_indices, return_counts=True)
        percentages = np.zeros(len(dominant_colors))
        percentages[unique] = counts / len(pixels)
        return percentages

    def create_color_bar_fixed_position(dominant_colors, percentages, frame_height, frame_width):
        dominant_colors_hsv = [rgb_to_hsv(color) for color in dominant_colors]
        dominant_colors_sorted = [color for _, color in
                                  sorted(zip(dominant_colors_hsv, dominant_colors), key=lambda x: x[0][0])]
        percentages_sorted = [x for _, x in sorted(zip(dominant_colors_hsv, percentages), key=lambda x: x[0][0])]

        color_bar = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        current_x = 0

        for i, color in enumerate(dominant_colors_sorted):
            if percentages_sorted[i] > 0:
                bar_width = int(percentages_sorted[i] * frame_width)
                next_x = current_x + bar_width

                next_color = dominant_colors_sorted[i + 1] if i < len(dominant_colors_sorted) - 1 else color

                # Smooth transition
                for x in range(current_x, next_x):
                    if next_x > current_x:  # Ensure there is a range to blend
                        blend_factor = (x - current_x) / (next_x - current_x)
                        blended_color = (1 - blend_factor) * color + blend_factor * next_color
                    else:
                        blended_color = color  # No blending if no range

                    color_bar[:, x] = blended_color

                current_x = next_x

        # Fill remaining width with last color if needed
        if current_x < frame_width:
            color_bar[:, current_x:] = dominant_colors_sorted[-1]

        return color_bar

    cap = cv2.VideoCapture(video_path)
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = original_frame_width // resize_factor
    frame_height = original_frame_height // resize_factor

    all_pixels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        all_pixels.append(resized_frame.reshape(-1, 3))
    all_pixels = np.vstack(all_pixels)
    dominant_colors = get_overall_dominant_colors(all_pixels, num_dominant_colors)
    cap.release()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_frame_width, original_frame_height))

    frame_index = 0
    color_percentages_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        percentages = calculate_color_percentages(resized_frame, dominant_colors)
        color_percentages_list.append(percentages)
        frame_index += 1

    smoothed_percentages = uniform_filter1d(np.array(color_percentages_list), size=smooth_factor, axis=0)
    cap.release()
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        color_percentages = smoothed_percentages[frame_index]
        color_bar = create_color_bar_fixed_position(dominant_colors, color_percentages, original_frame_height, original_frame_width)
        out.write(color_bar)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Video saved as {output_video_path}")
