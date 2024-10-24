import cv2
import os
import PySimpleGUI as sg
from processing import process_video, process_frame
from pixelate_processing import pixelate_video, pixelate_frame
from gpx_handler import process_gpx
from segmentation import run_segmentation

# Set the GUI color theme and custom font
sg.theme_background_color('#266850')
sg.theme_element_background_color('#266850')
sg.theme_text_color('white')

# Path to custom font (ensure the path is correct)
custom_font = ('assets/font/Trattatello.woff2', 12)

# Path to custom icon (ensure the path is correct and it's in .ico format)
custom_icon = 'assets/favicon.ico'

# Function to create the layout
def create_layout(process_mode='Cromaticon 3000', class_selection='Vegetation'):
    layout = [
        [sg.Text("4KHD Ultra Paesaggio Continuo", font=custom_font, background_color='#266850')],
        [sg.Text("Select Processing Mode", font=custom_font, background_color='#266850'),
         sg.Combo(['Cromaticon 3000', 'Piastrellificio.px', '.geopeg', 'Segmentatore Bugiardo Semantico'], default_value=process_mode, key="-PROCESS_MODE-",
                  enable_events=True, font=custom_font)],
        [sg.Text("Select Video File", font=custom_font, background_color='#266850'), sg.Input(key="-VIDEO-", enable_events=True), sg.FileBrowse(font=custom_font)],
        [sg.Text("Select GPX File", font=custom_font, background_color='#266850'), sg.Input(key="-GPX-", enable_events=True),
         sg.FileBrowse(file_types=(("GPX Files", "*.gpx"),), font=custom_font)],

        # Shared Resize Factor
        [sg.Text("Resize Factor", font=custom_font, background_color='#266850'),
         sg.Slider(range=(1, 10), orientation='h', key="-RESIZE-", default_value=5, background_color='#266850')],

        # Dominant Colors settings for Cromaticon 3000
        [sg.Text("Number of Dominant Colors", font=custom_font, background_color='#266850'),
         sg.Slider(range=(2, 20), orientation='h', key="-NUM_COLORS-", default_value=10, background_color='#266850')],
        [sg.Text("Smooth Factor", font=custom_font, background_color='#266850'),
         sg.Slider(range=(5, 20), orientation='h', key="-SMOOTH-", default_value=10, background_color='#266850')],

        # Pixelation settings for Piastrellificio.px
        [sg.Text("Number of Sectors", visible=False, font=custom_font, background_color='#266850'),
         sg.Slider(range=(1, 50), default_value=9, orientation="h", key="-NUM_SECTORS-", visible=False, background_color='#266850')],

        # Segmentation settings for Segmentatore Semantico Bugiardo
        [sg.Text("Select Class", font=custom_font, background_color='#266850'),
         sg.Combo(['Vegetation', 'Building', 'Sky'], default_value=class_selection, key="-CLASS_SELECTION-", enable_events=True, font=custom_font)],

        [sg.Image(key="-ORIGINAL_FRAME-"), sg.Image(key="-PROCESSED_FRAME-")],
        [sg.Radio("Use GPX", "RADIO1", key="-USE_GPX-", default=False, font=custom_font, background_color='#266850')],
        [sg.Text("Author:", font=custom_font, background_color='#266850'), sg.Input(key="-AUTHOR-", font=custom_font)],
        [sg.Text("Device:", font=custom_font, background_color='#266850'), sg.Input(key="-DEVICE-", font=custom_font)],
        [sg.Text("Category:", font=custom_font, background_color='#266850'), sg.Input(key="-CATEGORY-", font=custom_font)],
        [sg.Button("Preview", font=custom_font), sg.Button("Process", font=custom_font), sg.Button("Exit", font=custom_font)]
    ]

    # Modify layout based on process mode
    if process_mode == 'Cromaticon 3000':
        layout[7] = layout[8] = []
    elif process_mode == "Piastrellificio.px":
        # Show Pixelation settings, hide others
        layout[5] = layout[7] = layout[8] = []
        layout[6] = [sg.Text("Number of Sectors:", font=custom_font, background_color='#266850'),
                     sg.Slider(range=(1, 50), default_value=8, orientation="h", key="-NUM_SECTORS-", background_color='#266850')]
    elif process_mode == '.geopeg':
        layout[4] = layout[5] = layout[6] = layout[7] = layout[8] = []  # Hide settings for '.geopeg'
    elif process_mode == 'Segmentatore Bugiardo Semantico':
        layout[5] = layout[6] = []

    return layout

# Create the initial window with custom icon
layout = create_layout()
window = sg.Window("4KHD Ultra Paesaggio Continuo", layout, icon=custom_icon)

video_path, gpx_path = None, None
colors_path = "enet-cityscapes/enet-colors-vegetation.txt"  # Default to Vegetation

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break

    if event == "-VIDEO-":
        video_path = values["-VIDEO-"]
        # Automatically preview the video upon upload
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            original_frame = cv2.resize(frame, (320, 240))
            window["-ORIGINAL_FRAME-"].update(data=cv2.imencode('.png', original_frame)[1].tobytes())
        cap.release()

    if event == "-GPX-":
        gpx_path = values["-GPX-"]

    # Handle the class selection for segmentation
    if event == "-CLASS_SELECTION-":
        selected_class = values["-CLASS_SELECTION-"]
        if selected_class == "Vegetation":
            colors_path = "enet-cityscapes/enet-colors-vegetation.txt"
        elif selected_class == "Building":
            colors_path = "enet-cityscapes/enet-colors-building.txt"
        elif selected_class == "Sky":
            colors_path = "enet-cityscapes/enet-colors-sky.txt"

    # Handle the mode switch between Dominant Colors, Pixelation, Segmentation, and Original
    if event == "-PROCESS_MODE-":
        new_layout = create_layout(values["-PROCESS_MODE-"], values.get("-CLASS_SELECTION-", 'Vegetation'))
        window.close()  # Close the current window
        window = sg.Window("Video Processing", new_layout, icon=custom_icon)  # Create a new window with the new layout

    if event == "Preview" and video_path and values["-PROCESS_MODE-"] != ".geopeg":
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            original_frame = cv2.resize(frame, (320, 240))  # Resize for displaying in the GUI

            # Ensure only the original frame is displayed on the left
            window["-ORIGINAL_FRAME-"].update(data=cv2.imencode('.png', original_frame)[1].tobytes())

            # Process frame based on selected mode and display in the processed frame window
            if values["-PROCESS_MODE-"] != ".geopeg":
                resize_factor = int(values["-RESIZE-"])  # Unified Resize Factor

            if values["-PROCESS_MODE-"] == "Cromaticon 3000":
                num_colors = int(values["-NUM_COLORS-"])
                smooth_factor = int(values["-SMOOTH-"])  # Use the Smooth Factor for Cromaticon 3000
                processed_frame = process_frame(frame, num_colors, resize_factor, smooth_factor)

            elif values["-PROCESS_MODE-"] == "Piastrellificio.px":
                num_sectors = int(values["-NUM_SECTORS-"])
                processed_frame = pixelate_frame(frame, num_sectors, resize_factor)

            elif values["-PROCESS_MODE-"] == "Segmentatore Bugiardo Semantico":
                model_path = "enet-cityscapes/enet-model.net"
                classes_path = "enet-cityscapes/enet-classes.txt"
                processed_frame = run_segmentation(model_path, classes_path, colors_path, video_path, "", resize_factor,
                                                   preview=True)

            # If a processed frame was generated, display it in the processed frame window
            if processed_frame is not None:
                processed_frame_resized = cv2.resize(processed_frame, (320, 240))
                window["-PROCESSED_FRAME-"].update(data=cv2.imencode('.png', processed_frame_resized)[1].tobytes())

        cap.release()

    if event == "Process" and video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Determine the output folder based on the processing mode
        process_mode_name = values["-PROCESS_MODE-"].replace("<", "").replace(">",
                                                                              "")  # Remove invalid characters for folder names

        if values["-PROCESS_MODE-"] == ".geopeg":
            # Special case for .geopeg: no '_processed' suffix, just the video name as the folder
            output_video_folder = os.path.join("outputs", video_name)
            output_video_path = os.path.join(output_video_folder, f"{video_name}.mp4")
        else:
            # For other modes, use the normal naming convention
            output_video_folder = os.path.join("outputs", f"{video_name}_{process_mode_name}")
            output_video_path = os.path.join(output_video_folder, f"{video_name}_{process_mode_name}.mp4")

        # Ensure the output folder exists
        os.makedirs(output_video_folder, exist_ok=True)

        use_gpx = values["-USE_GPX-"]

        if values["-PROCESS_MODE-"] != ".geopeg":
            resize_factor = int(values["-RESIZE-"])  # Unified Resize Factor

        # Process the video based on the selected mode
        if values["-PROCESS_MODE-"] == "Cromaticon 3000":
            num_colors = int(values["-NUM_COLORS-"])
            smooth_factor = int(values["-SMOOTH-"])  # Use the Smooth Factor for Cromaticon 3000
            process_video(video_path, output_video_path, num_colors, resize_factor, smooth_factor)
            if use_gpx and gpx_path:
                gpx_data = process_gpx(gpx_path, output_video_path, values["-AUTHOR-"], values["-DEVICE-"],
                                       values["-CATEGORY-"], values["-PROCESS_MODE-"])

        elif values["-PROCESS_MODE-"] == "Piastrellificio.px":
            num_sectors = int(values["-NUM_SECTORS-"])
            pixelate_video(video_path, output_video_path, num_sectors, resize_factor)
            if use_gpx and gpx_path:
                gpx_data = process_gpx(gpx_path, output_video_path, values["-AUTHOR-"], values["-DEVICE-"],
                                       values["-CATEGORY-"], values["-PROCESS_MODE-"])

        elif values["-PROCESS_MODE-"] == "Segmentatore Bugiardo Semantico":
            model_path = "enet-cityscapes/enet-model.net"
            classes_path = "enet-cityscapes/enet-classes.txt"
            run_segmentation(model_path, classes_path, colors_path, video_path, output_video_path=output_video_path,
                             resize_factor=resize_factor)
            if use_gpx and gpx_path:
                gpx_data = process_gpx(gpx_path, output_video_path, values["-AUTHOR-"], values["-DEVICE-"],
                                       values["-CATEGORY-"], values["-PROCESS_MODE-"])

        elif values["-PROCESS_MODE-"] == ".geopeg":
            if use_gpx and gpx_path:
                gpx_data = process_gpx(gpx_path, video_path, values["-AUTHOR-"], values["-DEVICE-"],
                                       values["-CATEGORY-"], values["-PROCESS_MODE-"])

        sg.popup_no_buttons(f"Video processing ended! Siu siu siu :)", auto_close=True, no_titlebar=True,
                            background_color='#283b5b')
