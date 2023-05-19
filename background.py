from ast import Tuple
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import mit_semseg

import argparse


class BackgroundDivision:    
    
    def __init__(self):
        
        # Create the network
        net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='./ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='./ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.eval()
        self.segmentation_module.cuda()
        
        self.loader = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        
    def predict_segmentation(self, image : torch.Tensor) -> numpy.ndarray:
        
        dict_image = {'img_data': image[None].cuda()}
        seg_size = image.shape[1:]
        
        with torch.no_grad():
            scores = self.segmentation_module(dict_image, segSize=seg_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        
        return pred
    
    def from_pil_to_tensor(self, image: PIL.Image.Image) -> torch.Tensor:
        
        pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        
        return pil_to_tensor(image)
    
    def get_background_foreground(self, image : PIL.Image.Image, fill_black=False) -> Tuple(PIL.Image.Image, PIL.Image.Image):
    
        img = self.from_pil_to_tensor(image)
        pred = self.predict_segmentation(image=img)
        image_array = numpy.array(image)
        
        # Take the background
        background = self._mask_foreground_background(image_array, pred, mask_background=False, fill_black=fill_black)
        foreground = self._mask_foreground_background(image_array, pred, mask_background=True, fill_black=fill_black)
        
        return foreground, background
    
    def _mask_foreground_background(self, img:numpy.ndarray,  pred : numpy.ndarray, mask_background: bool, fill_black : bool) -> PIL.Image.Image:
        
        if mask_background:
            mask = pred == 2
        else:
            mask = pred != 2

        # Apply the mask
        image = img.copy()
        image [mask] = 0 if fill_black else 255

        return PIL.Image.fromarray(image)
    
    def close(self):
        self.segmentation_module = None
        self.loader = None
        torch.cuda.empty_cache()



if __name__ == '__main__':
    
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Background Division')

    # Add the filename argument
    parser.add_argument('--filename', type=str, help='Input image filename')

    # Add the background color choice argument
    parser.add_argument('--black-bg', action='store_true', help='Use black background (default is white)')

    # Parse the command-line arguments
    args = parser.parse_args()

    
    
    filename = args.filename
    use_black_background = args.black_bg
    
    # Load the image using PIL
    pil_image = PIL.Image.open(filename).convert('RGB')

    # Create an instance of BackgroundDivision
    bgd = BackgroundDivision()

    # Get the background and foreground images
    fg, bg = bgd.get_background_foreground(pil_image, fill_black=use_black_background)

    # Save the foreground image
    fg_filename = os.path.splitext(filename)[0] + '_fg_' + ('black' if use_black_background else 'white') + '.jpeg'
    fg.save(fg_filename)

    # Save the background image
    bg_filename = os.path.splitext(filename)[0] + '_bg_' + ('black' if use_black_background else 'white') + '.jpeg'
    bg.save(bg_filename)