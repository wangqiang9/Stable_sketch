# guidance text, sketch, images triple modals
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast, nn
from PIL import Image
import torchvision
from matplotlib import pyplot as plt
import numpy
import lpips
import os
from torchvision import transforms as tfms
from data.fscoco_dataload import TripleDataset, cycle
from torch.utils import data

# For video display:
from IPython.display import HTML
from base64 import b64encode

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True)

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True)

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# Using torchvision.transforms.ToTensor
to_tensor_tfm = tfms.ToTensor()

def pil_to_latent(input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
  return 0.18215 * latent.mode() # or .mean or .sample

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# loss functions
def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,-1, :, :] - 0.9).mean()
    return error

def image_loss(origin, images):
    error = torch.abs(origin.cpu() - images).mean()
    return error

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
def sketch_loss(origin, images):
    error = torch.abs(loss_fn_alex(origin.cpu(), images)).mean()
    return error


# settings
height = 256  # default height of Stable Diffusion
width = 256  # default width of Stable Diffusion
num_inference_steps = 200  # Number of denoising steps
guidance_scale = 8   # Scale for classifier-free guidance
generator = torch.manual_seed(32)  # Seed generator to create the inital latent noise
batch_size = 1
blue_loss_scale = 40  # @param
image_loss_scale = 1
sketch_loss_scale = 1

# Prep Scheduler
scheduler.set_timesteps(num_inference_steps)

# load data
photo_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/images"
sketch_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/raster_sketches"
text_root = "/root/sketchimage/fscoco-main/fscoco/fscoco/text"
ds = TripleDataset(photo_root=photo_root, sketch_root=sketch_root, text_root=text_root)
dl = cycle(data.DataLoader(ds, batch_size=1, drop_last=True, shuffle=True, pin_memory=True))

# index = 0
# while True:
#     index += 1
#     data = next(dl)
#     data_image = data["P"]
#     data_sketch = data["S"]
#     data_label = data["L"]
#     data_text = data["T"]
#     print(data_image.size(), data_sketch.size(), data_label, data_text)
#     # save_data = torch.cat((data_image, data_sketch), dim=0)
#     # utils.save_image(save_data, f"./test/{data_text}.png")

def train(save_path, dl, epochs=1000):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    txt_list = []
    transform_PIL = torchvision.transforms.ToPILImage()
    for epoch in range(epochs):
        data = next(dl)
        data_image = data["P"]
        data_sketch = data["S"]
        data_text = data["T"]

        # @title Store the predicted outputs and next frame for later viewing
        # prompt = 'Few sheep are eating grass on a mountain'  # fscoco/text/1/000000006336.txt
        prompt = data_text[0]
        txt_list.append(prompt)
        print(prompt)
        # Prep text
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        # And the uncond. input as before:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep latents
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * scheduler.sigmas[0]  # Need to scale to match k

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                #### ADDITIONAL GUIDANCE ###
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()

                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred

                # Decode to image space
                denoised_images = vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5  # (0, 1)

                # Calculate loss
                # loss = blue_loss(denoised_images) * blue_loss_scale
                loss = image_loss(denoised_images, data_image[0]) * image_loss_scale + sketch_loss(denoised_images, data_sketch[0]) * sketch_loss_scale
                # if i % 10 == 0:
                #     print(i, 'loss:', loss.item())
                print(i, 'loss:', loss.item())

                # Get gradient
                cond_grad = -torch.autograd.grad(loss, latents)[0]

                # Modify the latents based on this gradient
                latents = latents.detach() + cond_grad * sigma ** 2
                latents = latents.detach()

                ### And saving as before ###
                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred
                im_t0 = latents_to_pil(latents_x0)[0]

                # And the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
                im_next = latents_to_pil(latents)[0]

                if i == num_inference_steps - 1:
                    # save_data = torch.cat((data_sketch, data_image, torch.tensor(im_next)), dim=0)
                    # utils.save_image(save_data, f"{save_path}/{epoch}_{prompt}.png")
                    sketch_save = transform_PIL(data_sketch[0])
                    image_save = transform_PIL(data_image[0])
                    im = Image.new('RGB', (width * 3, height))
                    im.paste(sketch_save, (0, 0))
                    im.paste(image_save, (width * 1, 0))
                    im.paste(im_next, (width * 2, 0))
                    im.save(f'{save_path}/{epoch}_{prompt}.png')
                    print(f"save {save_path}/{epoch}_{prompt}.png!")
    with open(f'{save_path}/prompts.txt', 'w+') as f:
        for txt in txt_list:
            f.write(txt)
            f.write('\n')


if __name__ == "__main__":
    save_path = "./outputs/image_sketch_loss/9"
    data = dl
    epochs = 200
    train(save_path=save_path, dl=dl, epochs=epochs)
