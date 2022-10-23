import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

training_templates_smallest = [
    '{} {}',
]

reg_templates_smallest = [
    '{}',
]

training_captions = {
    "1": "a {} {} sitting on a couch smiling for the camera with a smile on her face and a necklace on her neck, by Rumiko Takahashi",
    "2": "a man and a {} {} standing next to a waterfall smiling for the camera with a smile on their face, by Junji Ito",
    "3": "a {} {} smiling and holding a plate of food in her hand and a microwave in the background with a sign on it, by Naoko Takeuchi",
    "4": "a {} {} with a smile on her face near the water and a beach with boats in the background and a pier in the foreground, by Naoko Takeuchi",
    "5": "a {} {} smiling at the camera, by Cindy Sherman",
    "6": "a {} {} with a blue scarf around her neck smiling at the camera with a sunflower in the background, by Naoko Takeuchi",
    "7": "a {} {} smiling for the camera with a car in the background and a parking lot behind her, by Junji Ito",
    "8": "a {} {} with a smile on her face taking a selfie with a camera phone in front of a wooden bench, by Studio Ghibli",
    "9": "a {} {} with glasses on her head and a wooden background behind her is smiling at the camera and has a camera strap around her neck, by Edith Lawrence",
    "10": "a {} {} with her arms up in the air and a rack of clothes behind her with her hands up, by Junji Ito",
    "11": "a {} {} standing in front of a waterfall and bridge with a train on it's tracks in the background, by Chen Daofu",
    "12": "a {} {} with a green shirt smiling at the camera, by Inio Asano",
    "13": "a {} {} with two bows on her head smiling at the camera with a smile on her face and a hair clip in her hair, by Rumiko Takahashi",
    "14": "a {} {} smiling for a picture at a sporting event in the sun, by Rumiko Takahashi",
    "15": "a {} {} wearing sunglasses and a flower in her hair smiling for the camera with a tree in the background, by Naoko Takeuchi",
    "16": "a {} {} with a smiley face on her head wearing a yellow hat and smiling at the camera, by Michelangelo Merisi Da Caravaggio",
    "17": "a {} {} with a smile on her face and a man in the background taking a picture of her with a cell phone, by Rumiko Takahashi",
    "18": "a {} {} with a smile on her face and a black jacket on her shoulders and a pink shirt on her shirt, by Rumiko Takahashi",
    "19": "a {} {} with a smile on her face and a pink shirt on her shirt is smiling at the camera, by Rumiko Takahashi",
    "20": "a {} {} with a smile on her face and a black shirt on her shirt is smiling and looking to the side, by Rumiko Takahashi",
    "21": "a {} {} is smiling in the snow with a scarf around her neck and a tree in the background with snow on it, by Rumiko Takahashi",
    "22": "a {} {} is smiling for the camera while sitting down with a man with glasses in the background and a red chair, by Junji Ito",
    "23": "a {} {} standing on a beach next to the ocean with a truck in the background and a person taking a picture, by Junji Ito",
}

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg = False,
                 max_images = None,
                 subject = 'sks',
                 ):

        self.subject = subject
        self.data_root = data_root
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        if max_images != None:
            self.num_images = min(self.num_images, max_images)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.reg = reg

    def setToken(token):
        self.token = token

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        imagepath = self.image_paths[i % self.num_images]
        image = Image.open(imagepath)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        name = os.path.splitext(os.path.basename(imagepath))[0]
        print(training_captions)
        if not self.reg and name in training_captions:
            text = training_captions[name].format(self.subject, placeholder_string)
        elif not self.reg:
            text = random.choice(training_templates_smallest).format(self.subject, placeholder_string)
        else:
            text = random.choice(reg_templates_smallest).format(placeholder_string)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example