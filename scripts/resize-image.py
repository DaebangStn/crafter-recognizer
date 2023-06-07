import cv2
import os
import glob

base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(base_dir)
# specify your input directory
input_dir = os.path.join(base_dir, 'img-raw')
# specify your output directory
output_dir = os.path.join(base_dir, 'img')

print('Input directory: ' + input_dir)

if not os.path.exists(input_dir):
    print('Input directory does not exist.')
    exit()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all .png files
files = glob.glob(os.path.join(input_dir, '*.png'))

print('Found ' + str(len(files)) + ' files.')

for file in files:
    img = cv2.imread(file)
    resized_img = cv2.resize(img, (66, 66))  # resize image to 66x66
    base_name = os.path.basename(file)  # get the original file name
    output_file_name = os.path.join(output_dir, base_name)  # create output file name
    cv2.imwrite(output_file_name, resized_img)  # save resized image
    print('Resized image: ' + output_file_name)
