# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Atak adwersarialny

# %% [markdown]
# ## Setup i instalacja

# %% [markdown]
# Pobieramy przykładowy obrazek pandy oraz listę etykiet ze zbioru imagenet

# %%
# !wget https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/640px-Grosser_Panda.JPG
# !wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json

# %%
# ! git clone https://github.com/Harry24k/adversarial-attacks-pytorch
# ! cd adversarial-attacks-pytorch
# ! git checkout 23620a694a3660e4f194c3e4d28992bced7785a1

# %% [markdown]
# Instalujemy zależności - tutaj torch i torchvision zostawiam do własnoręcznej instalacji, a informacje o użytej wersji pozostawiam poniżej.

# %%
# ! pip install pillow, matplotlib, 'transformers[torch]'

# %%
# torch==2.2.0+cu118, 
# torchvision==0.17.0+cu118, 
# torchattacks 23620a694a3660e4f194c3e4d28992bced7785a1
# PIL==10.2.0, 
# matplotlib==10.2.0
# transformers==4.37.2

# %%
import sys
sys.path.insert(0, 'adversarial-attacks-pytorch')

import torch
import torch.nn as nn
import torchattacks

# %% [markdown]
# ## Load model and data

# %%
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch
import torchvision.transforms as T

# %%
import json
class_idx = json.load(open("imagenet_class_index.json"))

# %%
HFmodel = ConvNextForImageClassification.from_pretrained("facebook/convnext-ground_truth_id-384-22k-1k", cache_dir="./cache").eval()
model = next(HFmodel.modules())


# %%

class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
        self.training = False

    def forward(self, x):
        x = self.model(x)
        return x.logits


# %%
nm = MyModel()


# %%
image_path = '640px-Grosser_Panda.JPG'
image = Image.open(image_path)
torch_original = T.ToTensor()(image)
torch_original = T.Resize([480, 640])(torch_original)

# %% [markdown]
# ## Adversarial Attack (Targeted)

# %%
res = nm(torch_original.unsqueeze(0))
base_prediction = torch.argmax(res).item()
class_name = class_idx[str(base_prediction)]
print(f"The base image prediction id is: {base_prediction}, class: {class_name}")

# %%
target_id = 368
ground_truth_id = 388
target_class_name = class_idx[str(target_id)]

# %%
print(f"The target prediction id is: {target_id}, class: {target_class_name}")

# %%
tf = lambda x: torch.tensor([target_id])
target_fun = lambda images, labels:tf(labels)

target = tf(ground_truth_id)
atk = torchattacks.PGD(nm, eps=1/255, alpha=1/255, steps=4)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
atk.set_mode_targeted_by_function(target_map_function=target_fun)

adv_images = atk(torch_original.unsqueeze(0), torch.tensor([ground_truth_id]))

logits = nm(adv_images)
prediction = torch.argmax(logits).item()
message = "CORRECT!" if target==prediction else "Wrong!"
print(f"{message} target:{target}, prediction:{prediction}, ground_truth_id was:{ground_truth_id}")

# %%
fig, axe = plt.subplots(1,2, figsize = (8,6))

crafted_image = adv_images[0].detach().permute(1,2,0).cpu().numpy()
original_image = torch_original.detach().permute(1,2,0).cpu().numpy()
axe[1].imshow(crafted_image)
crafted_class_name = class_idx[str(prediction)]
axe[1].set_title(f"Class: {crafted_class_name}")
axe[0].imshow(original_image)
axe[0].set_title(f"Class: {class_name}")

# %%
