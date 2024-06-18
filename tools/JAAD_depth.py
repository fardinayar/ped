from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import glob
import os
import tqdm
jaad_path = "/home/jvn-server/Desktop/zahraa/exp1/Pedestrian_Crossing_Intention_Prediction/JAAD"
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device='cuda')

# prepare image for the model

for image_name in tqdm.tqdm(glob.glob(os.path.join(jaad_path, "images/video*/*.png"))):
    image = Image.open(image_name)
    depth = pipe(image)["depth"]
    save_path = image_name.replace("images", "images_depth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth.save(save_path)