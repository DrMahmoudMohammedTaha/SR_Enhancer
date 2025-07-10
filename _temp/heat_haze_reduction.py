

import cv2

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Take the first frame as reference
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = []

    for i in range(1, n_frames):
        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        dx = flow[..., 0].mean()
        dy = flow[..., 1].mean()

        transforms.append((dx, dy))

        prev_gray = curr_gray

    # Apply the transforms
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    dx_sum, dy_sum = 0, 0
    for i in range(n_frames):
        success, frame = cap.read()
        if not success:
            break

        if i < len(transforms):
            dx_sum += transforms[i][0]
            dy_sum += transforms[i][1]

        m = np.float32([[1, 0, -dx_sum], [0, 1, -dy_sum]])
        stabilized_frame = cv2.warpAffine(frame, m, (w, h))
        out.write(stabilized_frame)

    cap.release()
    out.release()




import cv2
import numpy as np

def reduce_heat_haze(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Store previous N flows
    flow_history = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        flow_history.append(flow)
        if len(flow_history) > 5:
            flow_history.pop(0)

        # Median flow to reduce local distortions
        median_flow = np.median(np.array(flow_history), axis=0)

        # Warp current frame
        h_coords, w_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        map_x = (w_coords + median_flow[..., 0]).astype(np.float32)
        map_y = (h_coords + median_flow[..., 1]).astype(np.float32)

        corrected = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        out.write(corrected)

        prev_gray = gray

    cap.release()
    out.release()
