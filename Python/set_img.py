import pandas as pd
import numpy as np
from numpy.random import seed
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def img_name(dir_name):
    ''' 
    Get images filepath and store them in a list
    Required Parameter
        dir_name: Directory path
    Return
        None
    '''
    
    img_list = []
    for i in range(0, 7):
        sub_dir= dir_name + str(i) +'/'
        for sub in os.listdir(sub_dir):
            img_list.append(sub)
    return img_list

def create_img_dict(data_dir = '../SkinCancerClassifier/',
                    image_test_dir = '../SkinCancerClassifier/test_dir/', 
                    image_train_dir = '../SkinCancerClassifier/train_dir/', 
                    image_val_dir = '../SkinCancerClassifier/valid_dir/'):
    ''' 
    Creates raw dataframe and image dictionnary {key:image name, value:image path}
    
    Optional Parameters
        data_dir: Root directory for all files
        image_test_dir, image_train_dir, image_val_dir: 
        Directories for images in train, validation, and test
    Return
        Dataframe and image dictionnary
    '''
    
    # Create dataframe of raw data
    raw_metadata_df = pd.read_csv(data_dir + 'HAM10000_metadata.csv')
    # Create a combined list images in all directories with their full file/image path
    img_test_list = img_name(image_test_dir)
    img_test_list = [image_test_dir + img_name for img_name in img_test_list]
    img_train_list = img_name(image_train_dir)
    img_train_list = [image_train_dir + img_name for img_name in img_train_list]
    img_val_list = img_name(image_val_dir)
    img_val_list = [image_val_dir + img_name for img_name in img_val_list]
    all_img_list = img_test_list + img_train_list + img_val_list
    # Create a dictionnary that has the name of the image as its key and the image path as its value
    image_dict = {os.path.splitext(os.path.basename(img_name))[0]: img_name for img_name in all_img_list}
    
    return raw_metadata_df, image_dict

def set_df():
    '''
    This function creates 3 new columns for image file path (file_path), name of skin
    lesion (category_name), a number mapped to its name (category_id). For age column 
    with null value, it fills it with the mean age.
    
    Return
        Dataframe 
    '''
    
    # Create dictionary with the diagnostic categories of pigmented lesions
    lesion_cat_dict = {
        
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    # Get raw dataframe and the image dictionnary
    df, img_dict = create_img_dict()
    # Create new column file_path and use the image_id as the key of image_dict and map 
    # its corresponding value to get the path for the image
    df['file_path'] = df['image_id'].map(img_dict.get)
    # Create new column category_name and use dx as the key to lesion_cat_dict and map 
    # it to its corresponding value to get the lesion name
    df['category_name'] = df['dx'].map(lesion_cat_dict.get)
    # Create new column category_id and assign the integer codes 
    # of the category_name that were transformed into pandas categorical datatype
    df['category_id'] = pd.Categorical(df['category_name']).codes
    # Fill age null values by the mean age
    df.age.fillna(df.age.mean(), inplace=True)
    
    return df

def split_images(train_dir='../SkinCancerClassifier/train_dir/', 
                 test_dir='../SkinCancerClassifier/test_dir/', 
                 val_dir='../SkinCancerClassifier/valid_dir/', 
                 target_size=(90, 120)):
    ''' 
    Rescale, preprocess, apply image transformations 
    and load them in ImageDataGenerator
    
    Optional Parameters
        target_size: image pixel dimensions, by default 90x120
        train_dir, test_dir, val_dir: 
        Directories for images in train, validation, and test
    Return
        test_data, train_data, and valid_data
    '''

    # Rescale test, train, and validation images. Resize them to 90 x 120 pixels
    test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
                                                                       test_dir, 
                                                                       target_size=target_size, 
                                                                       batch_size = 2000, 
                                                                       seed = 1212
                                                                       ) 
    # Apply various transformations to the training data
    # to help the model generalize better on the test data
    train_data = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    rescale=1./255).flow_from_directory(
                                                                        train_dir, 
                                                                        target_size=target_size, 
                                                                        batch_size = 147, 
                                                                        seed = 1212
                                                                        ) 

    valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
                                                                        val_dir, 
                                                                        target_size=target_size,
                                                                        batch_size = 300,
                                                                        seed = 1212
                                                                        ) 
    
    return test_data, train_data, valid_data

def plot_confusion_matrix(te_label, y_pred):
    '''
    Plot confusion matrix and print classification report
    
    Parameters
        te_label: test images labels
        y_pred: predicted labels
    '''
    
    # Calculate Confusion Matrix
    cm = confusion_matrix(te_label, y_pred)
    df = set_df()
    # Set figure size and heatmap plot
    f = plt.figure(figsize=(10,10))
    ax= plt.subplot()
    labels = df.groupby('category_id').category_name.first().values
    sns.heatmap(cm, annot=True, ax = ax, vmax=100, cbar=False, cmap='Paired', mask=(cm==0), fmt=',.0f', linewidths=2, linecolor='grey', ); 

    # Set x, y labels and plot title
    ax.set_xlabel('Predicted labels', fontsize=16);
    ax.set_ylabel('True labels', labelpad=30, fontsize=16); 
    ax.set_title('Confusion Matrix', fontsize=18); 
    ax.xaxis.set_ticklabels(labels, rotation=90); 
    ax.yaxis.set_ticklabels(labels, rotation=0);
    ax.set_facecolor('white')
    
    report = classification_report(te_label, y_pred, target_names=df.groupby('category_id').category_name.first().values)
    print(report)
    
def plot_skin_images(img_list, title):
    '''
    Plot images of the skin lesions
    
    Parameters:
        img_list: list of images
        title: Plot title
    '''
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12 ,2))
    fig.suptitle(title)
    ax1.imshow(imread(img_list[0]))
    ax2.imshow(imread(img_list[1]))
    ax3.imshow(imread(img_list[2]))
    ax4.imshow(imread(img_list[3]))

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()

def display_images():
    '''
    Load 4 images per skin lesion category name
    
    Return
        image_dic: dictionnary of the images
    '''
    image_dic = {}
    actinic_keratosis_0 = ['../SkinCancerClassifier/valid_dir/0/ISIC_0026848.jpg', 
                       '../SkinCancerClassifier/valid_dir/0/ISIC_0026729.jpg', 
                       '../SkinCancerClassifier/valid_dir/0/ISIC_0030280.jpg', 
                       '../SkinCancerClassifier/valid_dir/0/ISIC_0024329.jpg']

    basal_carcinoma_1 = ['../SkinCancerClassifier/valid_dir/1/ISIC_0024666.jpg',
                        '../SkinCancerClassifier/valid_dir/1/ISIC_0025417.jpg',
                        '../SkinCancerClassifier/valid_dir/1/ISIC_0025046.jpg',
                        '../SkinCancerClassifier/valid_dir/1/ISIC_0026395.jpg']


    benign_keratosis_2 = ['../SkinCancerClassifier/valid_dir/2/ISIC_0024312.jpg',
                            '../SkinCancerClassifier/valid_dir/2/ISIC_0024477.jpg',
                            '../SkinCancerClassifier/valid_dir/2/ISIC_0024786.jpg',
                            '../SkinCancerClassifier/valid_dir/2/ISIC_0025157.jpg']


    dermatofibroma_3 = ['../SkinCancerClassifier/valid_dir/3/ISIC_0024386.jpg',
                        '../SkinCancerClassifier/valid_dir/3/ISIC_0027648.jpg',
                        '../SkinCancerClassifier/valid_dir/3/ISIC_0029891.jpg',
                        '../SkinCancerClassifier/valid_dir/3/ISIC_0030830.jpg']


    melanocytic_nv_4 = ['../SkinCancerClassifier/valid_dir/4/ISIC_0024340.jpg',
                        '../SkinCancerClassifier/valid_dir/4/ISIC_0025011.jpg',
                        '../SkinCancerClassifier/valid_dir/4/ISIC_0026010.jpg',
                        '../SkinCancerClassifier/valid_dir/4/ISIC_0027040.jpg']
        

    melanoma_5 = ['../SkinCancerClassifier/valid_dir/5/ISIC_0024459.jpg',
                '../SkinCancerClassifier/valid_dir/5/ISIC_0026131.jpg',
                '../SkinCancerClassifier/valid_dir/5/ISIC_0027100.jpg',
                '../SkinCancerClassifier/valid_dir/5/ISIC_0034275.jpg']


    vascular_lesions_6 = ['../SkinCancerClassifier/valid_dir/6/ISIC_0024904.jpg',
                        '../SkinCancerClassifier/valid_dir/6/ISIC_0031706.jpg',
                        '../SkinCancerClassifier/valid_dir/6/ISIC_0033123.jpg',
                        '../SkinCancerClassifier/valid_dir/6/ISIC_0033608.jpg']
    
    image_dic = {'Actinic_Keratosis_0': actinic_keratosis_0,
                 'Basal_Cell_Carcinoma_1': basal_carcinoma_1,
                 'Benign_Keratosis_2': benign_keratosis_2,
                 'Dermatofibroma_3': dermatofibroma_3,
                 'Melanocytic_Nevi_4': melanocytic_nv_4,
                 'Melanoma_5':  melanoma_5,
                 'Vascular_Lesions_6': vascular_lesions_6}
    
    return image_dic