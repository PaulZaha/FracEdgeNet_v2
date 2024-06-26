from PIL import Image, ImageFilter, ImageFile
import os
import shutil
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL.ImageFilter import (
   EDGE_ENHANCE_MORE
)

frac = 'K:/FracEdgeNet_v2/Dataset/FracAtlas/images/Fractured'

os.mkdir('K:/FracEdgeNet_v2/Dataset/FracAtlas/images/frac_augmented')
frac_augmented = 'K:/FracEdgeNet_v2/Dataset/FracAtlas/images/frac_augmented'


# #copy original fractures into frac_augmented folder
fracs = os.listdir(frac)
for datei in fracs:
    quellpfad = os.path.join(frac,datei)
    zielpfad = os.path.join(frac_augmented,datei)
    shutil.copy2(quellpfad,zielpfad)



# #alle fracs 0.1 reinzoomen
for filename in os.listdir(frac):
    img_path = os.path.join(frac, filename)
    filename = filename[:-4] + '_zoomed.jpg'
    img = Image.open(img_path)

    breite, höhe = img.size
    neue_breite = int(breite*0.9)
    neue_höhe = int(höhe*0.9)
    links = (breite - neue_breite) // 2
    oben = (höhe - neue_höhe) // 2
    rechts = links + neue_breite
    unten = oben + neue_höhe

    img_resized = img.crop((links,oben,rechts,unten)).resize((neue_breite,neue_höhe),Image.ANTIALIAS)
    output_path = os.path.join(frac_augmented,filename)
    img_resized.save(output_path)

# #alle bilder horizontal spiegeln
for filename in os.listdir(frac_augmented):
    img_path = os.path.join(frac_augmented, filename)
    filename = filename[:-4] + '_flipped.jpg'
    img = Image.open(img_path)

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)



    output_path = os.path.join(frac_augmented,filename)
    flipped_img.save(output_path)


#downsampling der non_fractures
sourcefolder = 'K:/FracEdgeNet_v2/Dataset/FracAtlas/images/Non_fractured'
os.mkdir('K:/FracEdgeNet_v2/Dataset/FracAtlas/images/non_fractured_downsampled')
destination_folder = 'K:/FracEdgeNet_v2/Dataset/FracAtlas/images/non_fractured_downsampled'
all_files = os.listdir(sourcefolder)
num_files_to_copy = 2868
files_to_copy = random.sample(all_files,num_files_to_copy)
for filename in files_to_copy:
    source_path = os.path.join(sourcefolder,filename)
    destination_path = os.path.join(destination_folder,filename)
    shutil.copy2(source_path,destination_path)

os.mkdir('K:/FracEdgeNet_v2/Dataset/FracAtlas/images/full_augmented')
all_images='K:/FracEdgeNet_v2/Dataset/FracAtlas/images/full_augmented'

fracs_aug = os.listdir(frac_augmented)
for datei in fracs_aug:
    quellpfad = os.path.join(frac_augmented,datei)
    zielpfad = os.path.join(all_images,datei)
    shutil.copy2(quellpfad,zielpfad)


non_fracs_downsampled = os.listdir(destination_folder)
for datei in non_fracs_downsampled:
    quellpfad = os.path.join(destination_folder,datei)
    zielpfad = os.path.join(all_images,datei)
    shutil.copy2(quellpfad,zielpfad)

# os.mkdir('K:/FracEdgeNet_v2/Dataset/FracAtlas/images/edge_detection')
# edge_detection='C:/Users/paulz/Documents/ProgrammingDigitalProjects_v2/Dataset_FracAtlas/images/edge_detection'
# #edge detection
# for filename in os.listdir(all_images):
#     print(filename)
#     img_path = os.path.join(all_images,filename)
#     img=Image.open(img_path)

#     img = img.filter(EDGE_ENHANCE_MORE)
    
#     output_path = os.path.join(edge_detection,filename)
#     img.save(output_path)