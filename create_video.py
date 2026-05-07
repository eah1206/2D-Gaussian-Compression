import argparse
import os
import cv2
import numpy as np

r"""
Example Usage: python3 create_video.py --path Splats/Patrick/residuals/10 --fps 24


"""


def load_folder(folder_name: str):
    """Returns the desired list of filenames in a specified folder"""
    image_extensions = {".png"}
    filenames = [
        name for name in sorted(os.listdir(folder_name))
        if os.path.splitext(name)[1].lower() in image_extensions
    ]

    if not filenames:
        raise FileNotFoundError(f"No image files found in {folder_name}")

    return filenames

def get_frames(folder_path: str, filenames: list):
    frames = []

    for name in filenames:
        full_path = os.path.join(folder_path, name)
        frame = cv2.imread(full_path)

        if frame is None:
            print(f"FAILED TO READ: {full_path}")
            raise ValueError(f"OpenCV could not read image: {full_path}")

        print(f"Loaded {name}: {frame.shape}")
        frames.append(frame)

    return frames

def load_reference(folder_name: str, splats: bool, num_splats: int) -> cv2.typing.MatLike:
    """Returns the first image frame of the folder specified."""
    ref_path = os.path.join('Splats' if splats else 'Output_Data', folder_name, 'frames')

    if splats:
        ref_path = os.path.join(ref_path, str(num_splats))

    filenames = load_folder(ref_path)
    ref_file = os.path.join(ref_path, filenames[0])

    ref_frame = cv2.imread(ref_file)

    if ref_frame is None:
        raise ValueError(f"OpenCV could not read reference image: {ref_file}")

    return ref_frame

def check_path(path: str):
    parts = os.path.normpath(path).split(os.sep)
    name = parts[1]
    is_splats = parts[0] == 'Splats'
    is_residual = parts[2] == 'residuals'
    num_splats = int(parts[-1]) if len(parts) == 4 else 0

    return name, is_residual, is_splats, num_splats


def parse_args():
    p = argparse.ArgumentParser(description="Create video clip from frames in a specified folder")
    p.add_argument("--path",      type=str,  required=True)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--method",    type=str,  default='old', choices=['old', 'new'])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name, is_residual, is_splats, num_splats = check_path(args.path)
    print(f'Name: {name}, is_residual: {is_residual}, is_splats: {is_splats}, num_splats: {num_splats}')
    filenames = load_folder(args.path)
    frames = get_frames(args.path, filenames)
    h, w, _ = frames[0].shape
    output_name = f"{name}_{'splats_'+str(num_splats) if is_splats else ''}_{'residuals' if is_residual else 'frames'}.mp4"
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
            if args.method == 'old':
                shifted = frame.astype(np.int16)
                diff = shifted - 128
                reconstructed = np.clip(prev_frame + diff, 0, 255).astype(np.uint8)
                output.write(reconstructed)
                prev_frame = reconstructed.astype(np.int16)


            # ----------------------------------------------------------------------------------------------------------------------------------
            # Other Implementaion. Residuals look weird, but reassembled seeingly losslessly. Uncomment and comment other implementation to run
            # ----------------------------------------------------------------------------------------------------------------------------------
            elif args.method == 'new':
                residual = frame.astype(np.uint8)
                reconstructed = (prev_frame.astype(np.int16) + residual.astype(np.int16)) % 256
                reconstructed = reconstructed.astype(np.uint8)
                output.write(reconstructed)
                prev_frame = reconstructed


    output.release()