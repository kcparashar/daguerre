# CS194-26: Computational Photography
# Final Project: Faking Tilt Shifts to Make Image Miniatures
# main.py
# Krishna Parashar

from miniatures import *

### Configurations
source_dir = 'data/'
dest_dir = 'results/'
extension = '.jpg'
attribute = "_mini"
levels = 1
img1_side = "top"
blurred_img_side = "bottom"

img_name = "mlk"
img1_name = "mlk.jpg"
blurred_img_name = "mlk_blurred.jpg"
mask_name = "mlk_mask.jpg"

### Load Images and Mask (And Create Blurred Image)
img1 = cv2.imread(source_dir + img1_name, cv2.IMREAD_COLOR)
blurred_img = cv2.bilateralFilter(img1, 9, 150, 150)
blurred_img = cv2.blur(img1, (15, 15))
write_image(source_dir, img_name, blurred_img, "_blurred", extension)
blurred_img = cv2.imread(source_dir + blurred_img_name, cv2.IMREAD_COLOR)
mask = cv2.imread(source_dir + mask_name, cv2.IMREAD_COLOR)
mask = mask[:, :, 0]

### Compute Stacks for Each Image and Mask
img1_stack  = laplacian_stack(img1, levels, (gaussian_stack(img1, levels)))
blurred_img_stack  = laplacian_stack(blurred_img, levels, (gaussian_stack(blurred_img, levels)))
mask_stack  = gaussian_stack(mask, levels)

### Apply Mask to Each Image in Image Stacks
masked_img1 = mask_imgs(img1_stack, mask_stack, img1_side)
masked_blurred_img = mask_imgs(blurred_img_stack, mask_stack, blurred_img_side)

### Sum Up Masked Images in Stacks to Create One Image
full_image = []
for img_1, img_2 in zip(masked_img1, masked_blurred_img):
    full_image.append(img_1 + img_2)

### Multi-resolution Blend Full Image using Mask
blended_img = blend(img1, blurred_img, img1_side, blurred_img_side, full_image, mask, levels)

### Enhance Image Color and Contrast
blended_img = Image.fromarray(np.uint8(blended_img))
blended_img = ImageEnhance.Color(blended_img).enhance(1.9)
blended_img = ImageEnhance.Contrast(blended_img).enhance(1.1)
blended_img = np.array(blended_img)

### Write Image out to File
write_image(dest_dir, img_name, blended_img, attribute, extension)


