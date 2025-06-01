from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)

import subprocess

def run_ns_train(data_path, max_iterations=30000, eval_mode="interval", eval_interval=8):
    """
    Run the ns-train command with specified parameters.

    Args:
        data_path (str): Path to the dataset.
        max_iterations (int): Maximum number of iterations for training.
        eval_mode (str): Evaluation mode.
        eval_interval (int): Interval for evaluation.
    """
    command = [
        "ns-train", "nerfacto",
        "--data", data_path,
        "--max-num-iterations", str(max_iterations),
        "--vis", "wandb",
        "nerfstudio-data",
        "--eval-mode", eval_mode,
        "--eval-interval", str(eval_interval),
    ]

    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print output and errors
    print("Output:\n", process.stdout)
    print("Errors:\n", process.stderr)

    # Check for errors
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}.")
        return False
    return True

Datasets = {
    # 'YanshanPark':{
    #     "output_dir":'/media/shuo/T7/duslam/video_images/china_classical_park_512/nerf_eval',
    # },
    # 'TaicangPark':{
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_taicang_LJf7LKLvmUc/image/512/nerf_eval',
    # },

    # 'Upplasa':{
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/512/nerf_eval',
    # },
    # 'Lund':{
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/512/nerf_eval',
    # },

    # 'Helsi':{
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/nerf_eval',
    # },
    'Helsi2':{
        "output_dir":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/nerf_eval_2',
    },


}

for dataset in Datasets:
    print(f"Training for dataset: {dataset}")
    dirs = Datasets[dataset]['output_dir']
    # list all dir in the dirs
    import os
    for dir in os.listdir(dirs):
        dir_path = os.path.join(dirs, dir)
        for _dir in os.listdir(dir_path):
            data_path = os.path.join(dir_path, _dir)
            print(f"Training for data path: {data_path}")
            success = run_ns_train(data_path)
            if not success:
                continue
            print(f"Training completed for data path: {data_path}")

    # success = run_ns_train(dataset)
    # if not success:
    #     continue
    # print(f"Training completed for dataset: {dataset}")