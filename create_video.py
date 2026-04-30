import argparse
import os
import cv2
import numpy as np

r"""
Example Usage: python create_video.py --path Splats\Patrick\frames\100 --fps 24

"""


def load_folder(folder_name: str):
    """Returns the desired list of filenames in a specified folder"""
    filenames = sorted(os.listdir(folder_name))
    return filenames

def get_frames(folder_path: str, filenames: list):
    frames = []
    for name in filenames:
        frames.append(cv2.imread(os.path.join(folder_path, name)))
    return frames

def load_reference(folder_name: str, splats: bool, num_splats: int)-> cv2.typing.MatLike:
    """Returns the first frame of the folder specified"""
    ref_path = os.path.join('Splats' if splats else 'Output_Data', folder_name, 'frames')
    if splats:
        ref_path = os.path.join(ref_path, str(num_splats))
    return cv2.imread(os.path.join(ref_path, sorted(os.listdir(ref_path))[0])) # type: ignore

def check_path(path: str):
    parts = os.path.normpath(path).split(os.sep)
    name = parts[1]
    is_splats = parts[0] == 'Splats'
    is_residual = parts[2] == 'residuals'
    num_splats = int(parts[-1]) if len(parts) == 4 else 0

    return name, is_residual, is_splats, num_splats


def parse_args():
    p = argparse.ArgumentParser(description="Create video clip from frames in a specified folder")
    p.add_argument("--path",      type=str, required=True)
    p.add_argument("--fps",         type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name, is_residual, is_splats, num_splats = check_path(args.path)
    print(f'Name: {name}, is_residual: {is_residual}, is_splats: {is_splats}, num_splats: {num_splats}')
    filenames = load_folder(args.path)
    frames = get_frames(args.path, filenames)
    h, w, _ = frames[0].shape
    output_name = f'{name}_{'splats_'+str(num_splats) if is_splats else ''}_{'residuals' if is_residual else 'frames'}.mp4'
    output = cv2.VideoWriter(os.path.join('Output_Video', output_name), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)) # type: ignore
    if not is_residual: # This is from a frames folder
        for frame in frames:
            output.write(frame)
    else:
        ref_frame = load_reference(name, is_splats, num_splats)
        output.write(ref_frame)
        prev_frame = ref_frame.astype(np.int16)
        for frame in frames:

            # -------------------------------------------------------------------------------------------------------------
            # Original implementation, slight ghosting when reassembled. Uncomment and comment other implementation to run
            # -------------------------------------------------------------------------------------------------------------
            shifted = frame.astype(np.int16)
            diff = shifted - 128
            reconstructed = np.clip(prev_frame + diff, 0, 255).astype(np.uint8)

            # ----------------------------------------------------------------------------------------------------------------------------------
            # Other Implementaion. Residuals look weird, but reassembled seeingly losslessly. Uncomment and comment other implementation to run
            # ----------------------------------------------------------------------------------------------------------------------------------
            #residual = frame.astype(np.uint8)
            #reconstructed = (prev_frame.astype(np.uint16) + residual.astype(np.uint16)).astype(np.uint8)

            output.write(reconstructed)
            prev_frame = reconstructed.astype(np.int16)
    output.release()