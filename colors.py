import os
import cv2
import pkg_resources
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

class color_coherence:
    def __init__(self):
        self.matcher = ColorMatcher()

    def maintain_colors(self, image, sample_image, sample_alpha, mode, idx, cc_mix_outdir=None, timestring=None, suppress_console=False, console_msg=""):
        ''' main function for color_coherence, assumes BGR array for image and color match sample '''
        sample_alpha = min(max(sample_alpha, 0), 1)
        
        # Ensure sample image is resized to match the dimensions of the input image
        sample_image = cv2.resize(sample_image, (image.shape[1], image.shape[0]))

        matched = np.copy(image.astype(np.uint8))
        sample = np.copy(sample_image.astype(np.uint8))

        sample_image = cv2.resize(sample_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        skimage_version = pkg_resources.get_distribution('scikit-image').version
        is_skimage_v20_or_higher = pkg_resources.parse_version(skimage_version) >= pkg_resources.parse_version('0.20.0')
        match_histograms_kwargs = {'channel_axis': -1} if is_skimage_v20_or_higher else {'multichannel': True}

        if mode in ['HM', 'Reinhard', 'MVGD', 'MKL', 'HM-MVGD-HM', 'HM-MKL-HM']:
            matched = self.matcher.transfer(src=image, ref=sample, method=mode.lower())
            matched = Normalizer(matched).uint8_norm()
        elif mode == 'RGB':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2RGB)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)
        elif mode == 'HSV':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2HSV)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_HSV2BGR)
        else:  # mode == 'LAB' (default)
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2LAB)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

        print(f"Color coherence {mode} applied to frame {idx} | {console_msg}")

        return matched

def optimized_pixel_diffusion_blend(image1, image2, alpha, cc_mix_outdir=None, timestring=None, idx=None):
    alpha = min(max(alpha, 0), 1)
    beta = 1 - alpha
    random_matrix = np.random.uniform(0, 1, image1.shape[:2])
    alpha_mask = random_matrix < alpha
    beta_mask = (random_matrix >= alpha) & (random_matrix < alpha + beta)
    result = np.copy(image1)
    result[alpha_mask] = image1[alpha_mask]
    result[beta_mask] = image2[beta_mask]

    if cc_mix_outdir is not None and timestring is not None and idx is not None:
        full_filepath = os.path.join(cc_mix_outdir, f'{timestring}_{idx:09}.jpg') 
        cv2.imwrite(full_filepath, result.astype(np.uint8))

    return result

def pil_to_cv2(image):
    '''Convert PIL Image to OpenCV image format (BGR)'''
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    '''Convert OpenCV image format (BGR) to PIL Image'''
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Example usage
def process_image(input_image_path, sample_image_path, sample_alpha, mode, idx):
    # Load the input PIL image and sample image
    input_pil_image = Image.open(input_image_path)
    sample_pil_image = Image.open(sample_image_path)

    # Convert PIL images to OpenCV images (BGR format)
    input_cv2_image = pil_to_cv2(input_pil_image)
    sample_cv2_image = pil_to_cv2(sample_pil_image)

    # Initialize color_coherence object
    cc = color_coherence()

    # Perform color coherence operation
    matched_cv2_image = cc.maintain_colors(input_cv2_image, sample_cv2_image, sample_alpha, mode, idx)

    # Convert the matched OpenCV image back to PIL image
    output_pil_image = cv2_to_pil(matched_cv2_image)

    return output_pil_image

# Example input paths
# input_image_path = 'input_image.jpg'
# sample_image_path = 'sample_image.jpg'
# sample_alpha = 0.5
# mode = 'LAB'
# idx = 0

# # Process and save the output image
# output_image = process_image(input_image_path, sample_image_path, sample_alpha, mode, idx)
# output_image.save('output_image.jpg')
