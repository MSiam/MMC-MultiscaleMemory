import math

def sample_frame_indices(stride, frame_index, num_frames, vid_len):
    frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(num_frames) / 2)*stride,
                                                            frame_index + math.ceil(float(num_frames) / 2)*stride,
                                                            stride)]
    return frame_indices

