import os
import shutil

ORG_DATA_PATH = "/home/hacho/@WSL-Projects/SLP2022/SLP/danaLab"
TYPE = ["IR", "RGB"]
COVER = ["uncover", "cover1", "cover2"]  # cover1: 얇은 이불, cover2: 두꺼운 이불


def copyfile_mkdirs(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


for MODALITY in TYPE:
    for cover in COVER:
        # train
        for i in range(0, 70):
            path_h = f"{ORG_DATA_PATH}/00{i + 1:03d}/{MODALITY}/{cover}"
            for j in range(0, 15):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/train/0/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)
            for j in range(15, 45):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/train/1/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)

        # val
        for i in range(70, 90):
            path_h = f"{ORG_DATA_PATH}/00{i + 1:03d}/{MODALITY}/{cover}"
            for j in range(0, 15):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/val/0/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)
            for j in range(15, 45):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/val/1/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)

        # test
        for i in range(90, 100):
            path_h = f"{ORG_DATA_PATH}/00{i + 1:03d}/{MODALITY}/{cover}"
            for j in range(0, 15):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/test/0/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)
            for j in range(15, 45):
                path_img = f"{path_h}/image_0000{j + 1:02d}.png"
                new_path = f"/home/hacho/@WSL-Projects/slp/data/{MODALITY}/{cover}/test/1/{i + 1:03d}_{j + 1:02d}.png"
                copyfile_mkdirs(path_img, new_path)
