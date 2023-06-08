from PIL import Image
import os
import glob

character_name = ['arrow', 'cow', 'plant', 'player', 'skeleton', 'zombie']
background_name = ['grass', 'sand', 'path']

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

files_character = []
files_background = []

for file in files:
    img = Image.open(file)
    resized_img = img.resize((66, 66))  # resize image to 66x66
    base_name = os.path.basename(file)  # get the original file name
    output_file_name = os.path.join(output_dir, base_name)  # create output file name
    resized_img.save(output_file_name)  # save resized image
    print('Resized image: ' + output_file_name)

    image_name = os.path.splitext(base_name)[0]
    if image_name in character_name:
        files_character.append(output_file_name)
    elif image_name in background_name:
        files_background.append(output_file_name)

for file_character in files_character:
    for file_background in files_background:
        img_character = Image.open(file_character).convert('RGBA')
        img_background = Image.open(file_background).convert('RGBA')
        img_background.paste(img_character, (0, 0), img_character)

        base_name1 = os.path.basename(file_character)
        base_name1 = os.path.splitext(base_name1)[0]
        base_name2 = os.path.basename(file_background)
        base_name2 = os.path.splitext(base_name2)[0]

        output_file_name = os.path.join(output_dir, base_name1 + '-' + base_name2 + '.png')
        img_background.save(output_file_name)
        print('Combined image: ' + output_file_name)

