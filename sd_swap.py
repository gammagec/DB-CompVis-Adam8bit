import cv2
import numpy as numpy
import mediapipe as mp
from resizeimage import resizeimage
import PIL
from PIL import Image
import matplotlib.image
import os, argparse
import yaml, torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm, trange
from einops import rearrange, repeat

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def convert_img(pil_img):
	image = pil_img.convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h})")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = numpy.array(image).astype(numpy.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_loc",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--video_in",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="?",
    )
    parser.add_argument(
        "--prompt",
        nargs="?"
    )
    parser.add_argument(
        "--write_pics",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--show_preview",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--include_originals",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--write_pics_no_face_frames",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--no-all_faces",
        dest="all_faces",
        action='store_false',
        default=True,
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--stop_after",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
    )

    opt = parser.parse_args()
    prompt = opt.prompt
    steps = opt.steps
    out_path = opt.out_dir
    fps = opt.fps
    write_pics = opt.write_pics
    frame_skip = opt.frame_skip
    show_preview = opt.show_preview
    skip_frames = opt.skip_frames
    stop_after = opt.stop_after
    include_originals = opt.include_originals
    expand_up = opt.expand
    expand = opt.expand
    all_faces = opt.all_faces
    write_pics_no_face_frames = opt.write_pics_no_face_frames
    scale = opt.scale
    strength = opt.strength

    cap = cv2.VideoCapture(opt.video_in)

    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    if not write_pics:
        print(f"creating video with size {frame_size} and frame rate {fps}")	
        out_vid = cv2.VideoWriter(f'{out_path}/out.mp4',
            cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt_loc)
    print("model loaded")

    class MediaPipeFaceDetect:
        def __init__(self):
            mp_face_detection = mp.solutions.face_detection
            self.detector = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence=0.2)

        def get_faces(self, frame):
            image_height, image_width, _ = frame.shape
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.detector.process(imgRGB)

            faces = []
            if results.detections: 
                # Iterate over the found faces.
                for face_no, face in enumerate(results.detections):			
                    face_bbox = face.location_data.relative_bounding_box
                    
                    x1 = int(face_bbox.xmin * image_width)
                    y1 = int(face_bbox.ymin * image_height)
                    w = int(face_bbox.width * image_width)
                    h = int(face_bbox.height * image_height)					

                    faces.append((x1, y1, w, h))
            print(f'got {len(faces)} faces')
            return faces

    count = 0
    num = 0

    faceDetect = MediaPipeFaceDetect()

    while (cap.isOpened()):	
        # Capture frame-by-frame	
        ret, frame = cap.read()		
        if not ret:
            break
        
        if count < skip_frames or count % frame_skip > 0:
            count += 1
            continue

        if stop_after > 0:
            print(f"processing {num} of {stop_after}")

        print(f'processing frame {count}')

        if show_preview:
            cv2.imshow('Original', frame)

        faces = faceDetect.get_faces(frame)

        original_image = frame.copy()

        count += 1		
        
        if len(faces) > 0:
            num += 1
            original_pil = Image.fromarray(original_image)

            #largest_face = None
            #face_area = 0
            #for face in faces:
            #    (x, y, w, h) = face
            #    if w * h > face_area:
            #        face_area = w * h
            #        largest_face = face
            #face = largest_face

            for face in faces:		
                (x, y, w, h) = face
                x -= expand
                y -= expand_up
                w += expand * 2
                h += expand_up + expand
                box = (x, y, x + w, y + h)
                print(f'box is {w}, {h}')
                im_pil = Image.fromarray(frame).convert("RGB").crop(box = box)
                im_pil_src = im_pil.copy()			
                (width, height) = im_pil.size
                if height > width:
                    new_width = 512
                    new_height = int((new_width / width) * height)
                else:
                    new_height = 512
                    new_width = int((new_height / height) * width)
                im_pil = im_pil.resize((new_width, new_height))
                im_pil = resizeimage.resize_cover(im_pil, [512, 512])
                print(f'size is {im_pil.size}')
                im_np = numpy.asarray(im_pil)		
                if show_preview:
                    cv2.imshow('Face', im_np)
                cv2.imwrite(f'{out_path}/in{count}_out.png', im_np)						
                output_image = run_img(model, im_pil, prompt, 
                    steps = steps, scale = scale, strength = strength)			
                # test write output image
                output_image_np = numpy.asarray(output_image)
                ouput_image_np = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)			
                cv2.imwrite(f'{out_path}/out_{count}_out.png', output_image_np)						
                if h > w:
                    new_h = h
                    new_w = int((new_h / 512) * 512)
                else:
                    new_w = w
                    new_h = int((new_w / 512) * 512)
                out_rs_w = output_image.resize((new_w, new_h))
                print(f'new {new_w} {new_h}')
                print(f'after resize_width {out_rs_w.width} {out_rs_w.height}')
                output_back = resizeimage.resize_crop(out_rs_w, [w, h])
                original_pil.paste(output_back, (x, y))
                if include_originals:
                    original_pil.paste(im_pil_src, (x + w, y))

            original_np = numpy.asarray(original_pil)
            if show_preview:
                cv2.imshow('SD Merged', original_np)
            if write_pics:
                # write pic			
                original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)			
                matplotlib.image.imsave(f'{out_path}/out_{count}.png', original_np)						
            else:
                out_vid.write(original_np)		
        else:		
            if not write_pics:					
                out_vid.write(frame)
            elif write_pics_no_face_frames:
                original_np = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)			
                matplotlib.image.imsave(f'{out_path}/out_{count}.png', original_np)										


        if stop_after > 0 and num > stop_after:
            break

        # define q as the exit button
        # only with 'show_preview' enabled
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
     
    # release the video capture object
    cap.release()
    if not write_pics:					
        out_vid.release()

    # Closes all the windows currently opened.
    cv2.destroyAllWindows()

def run_img(model, image, prompt, steps, scale, strength):

    device = torch.device("cuda")
    model = model.to(device)
    sampler = DDIMSampler(model)
    n_samples = 1
    n_iter = 1
    W = 512
    H = 512
    ddim_steps = steps
    ddim_eta = 0.0
    batch_size = 1

    image = convert_img(image)
    init_image = image.to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    data = [batch_size * [prompt]]
    #init_image.to(device)


    with torch.no_grad():
        with model.ema_scope():
            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        return Image.fromarray(x_sample.astype(numpy.uint8))
    return None

if __name__ == "__main__":
    main()