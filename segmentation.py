import numpy as np
import cv2

def run_segmentation(model_path, classes_path, colors_path, video_path, output_video_path=None, resize_factor=1, show=False, preview=False):
    # Load class labels
    CLASSES = open(classes_path).read().strip().split("\n")

    # If a colors file was supplied, load it
    if colors_path:
        COLORS = open(colors_path).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")
    else:
        # Randomly generate RGB colors for each class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

    # Load the deep learning segmentation model
    print("[INFO] Loading model...")
    net = cv2.dnn.readNet(model_path)

    # Initialize video stream
    vs = cv2.VideoCapture(video_path)

    # Retrieve the input video's FPS and original dimensions
    fps = vs.get(cv2.CAP_PROP_FPS)
    orig_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Full video generation: initialize the writer if we're not in preview mode
    writer = None
    if not preview and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "mp4v" codec for MP4 files
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height), True)

    frame_number = 0

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

        # Prepare the frame for segmentation
        blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        # Get the number of classes and the dimensions of the mask
        (numClasses, height, width) = output.shape[1:4]

        # Find the class ID with the largest probability for each pixel
        classMap = np.argmax(output[0], axis=0)

        # Map the class IDs to colors
        mask = COLORS[classMap]

        # Resize the mask back to the original frame size
        mask_resized = cv2.resize(mask, (resized_frame.shape[1], resized_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_final = cv2.resize(mask_resized, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

        # If in preview mode, return the processed single frame
        if preview:
            vs.release()
            return mask_final  # Return the frame for previewing

        # Otherwise, write the full video
        if writer is not None:
            writer.write(mask_final)

        # Optionally display the output frame in real-time
        if show:
            cv2.imshow("Frame", mask_final)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_number += 1

    # Cleanup
    print("[INFO] Cleaning up...")
    vs.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return None
