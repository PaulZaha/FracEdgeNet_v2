import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pandas as pd
import xml.etree.ElementTree as ET

def showimage(name,general_info_df):
    """
    Shows images using matplotlib package. Args[name: 'image_id.jpg']
    """

    #Navigiert path in Bilder Ordner
    os.chdir(os.path.join(os.getcwd(),'FracAtlas','images','full_augmented'))

    
    #Plots erstellen, Laden  des Bildes
    fig,ax = plt.subplots()
    ax.imshow(mpimg.imread(name),cmap='gray')
    plt.axis('off')
    #cwd zurück auf Standard-Ordner setzen
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
    #Falls Bruch, wird boundingbox auf plot gelegt
    if general_info_df.loc[general_info_df['image_id'] == name, 'fractured'].values == 1:
        rectangle = boundingbox(name,fig,ax)
        ax.add_patch(rectangle)

    #zeigt PLot an
    plt.show()
    
def boundingbox(name,fig,ax):
    """
    Shows boundingbox in showimage() if the picture is a fracture.#
    DANGER: Does not work for augmented fratures.
    """
    #Pathing in xml Ordner
    path = os.path.join(os.getcwd(),'FracAtlas','Annotations','PASCAL VOC')
    os.chdir(path)

    #Tree initialisieren
    tree = ET.parse(name[:-3]+'xml')
    root = tree.getroot()

    #Pathing zurück auf Standard-Ordner
    os.chdir(os.path.join(os.getcwd(),'..','..','..'))
    
    #Werte aus bndbox element aus XML ziehen
    values = []
    for x in root[5][4]:
        values.append(int(x.text))
    
    #reassign value 2 und 3, da wir width & height brauchen statt 4 koordinaten für patches.rectangle
    values[2] = values[2]-values[0]
    values[3] = values[3]-values[1]

    #Rechteck wird festgelegt
    bounding_box = patches.Rectangle([values[0],values[1]],width=values[2],height=values[3],linewidth=1,edgecolor='r',facecolor='none')
    
    return bounding_box



def main():
    os.chdir(os.path.join(os.getcwd(),'Dataset','FracAtlas'))
    #Einlesen der dataset.csv Datei
    general_info_df = pd.read_csv('dataset_preaugmented.csv')
    #cwd zurück zu ProgrammingDigitalProjects setzen
    os.chdir(os.path.join(os.getcwd(),'..'))

    showimage('IMG0000143.jpg',general_info_df=general_info_df)


if __name__ == "__main__":
    main()