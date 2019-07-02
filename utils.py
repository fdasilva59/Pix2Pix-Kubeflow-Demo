# ******************************************************************
#                 UTILS FUNCTIONS (for local use)
#  These functions helps to test the code  executed locally 
#  (Not for using within a Kubeflow Pipeline)
# ******************************************************************

from distutils.version import LooseVersion
from platform import python_version  
import glob
import os

# Deep Learning libraries
import numpy as np
import tensorflow as tf
from tensorflow import test
from tensorflow.python.client import device_lib

# Check if PIL package is available and eventually install it
try:
    from PIL import Image

except ImportError as e:
    print("[INFO] Installing Pillow" )
    from pip._internal import main as pip
    # packages to install
    packages_to_install = ["Pillow"]

    for package in packages_to_install:
        pip(["install",  "--disable-pip-version-check", package])  #"--quiet",

finally:
    from PIL import Image
    

# configure Matplotlib
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.style.use('default')
np.set_printoptions(precision=3, linewidth=120)


def check_config():
    ''' 
    Check if the local configuration met the technical requirements
    to execute this implemenattion of Pix2Pix
    '''          
    try:
        # Check Tensorflow version
        assert LooseVersion(tf.__version__) in [LooseVersion('1.12.0')], \
        'This project was designed and tested using TensorFlow version 1.12.0  You are using {} Please consider updating your environment by using  \n !pip install --ignore-installed tensorflow==1.12.0 or !pip install --ignore-installed tensorflow-gpu==1.12.0 ' \
                    .format(tf.__version__)

        # Display Information
        print("Python:{} ".format(python_version()))
        print("Tensorflow:{} ".format(tf.__version__))
        print("tf.keras:{} ".format(tf.keras.__version__))
        print(" Keras image data format : {}".format(tf.keras.backend.image_data_format()))
        if tf.test.is_gpu_available():
            print("GPU:{} ".format(tf.test.gpu_device_name()))
            devices=device_lib.list_local_devices()
            gpu=[(x.physical_device_desc) for x in devices if x.device_type == 'GPU']
            print("GPU details:", gpu) 
        else:
            print("No GPU available")

    except Exception as exc:
        print('/!\/!\/!\ WARNING /!\/!\/!\ \n\n' + str(exc))


# def display_image_array(img_list):
#     ''' 
#     Convert image arrays into an image format and display it
    
#     Args:
#         img_list (Numpy array) : list of image arrays                                  
#     '''        
#     plt.figure(figsize=(7,7))
#     for i, im in enumerate(img_list):
#         # Convert image array into Image format
#         img = np.squeeze(im, axis=0 ) # Drop batch dimension 
#         img = (img * 127.5) + 127.5   # Reverse normalisation
#         img = img.astype(int)            
#         img = np.clip(img, 0, 255)
#         ax = plt.subplot(121+i)
#         ax.imshow(img)
#     plt.show()
#     plt.close()


def display_training_examples(path_to_images, display_max_examples=3, reverse=True):
    '''
    Display a few image examples from the the Dataset Input/target 
    and associated Generated images from the Pix2Pix GAN generator
    
    Args:
        path_to_ouputs (str) : path to the Pix2Pix outputs
        display_max_examples (int) : Number of samples to display (0 to display all)
        reverse (bool) : False = oldest images  first and vice-versa
    '''     
    
    # During the training, images are saved to disk in the following order :
    # img_a, img_b and fake_b. Thus we can compute list of picture files from latest 
    # to oldest or vice-versa 
    images_a = [s for s in glob.glob(path_to_images + "/img_a*" ) if os.path.isfile(s)]
    images_a.sort(key=lambda s: os.path.getmtime(s), reverse=reverse)

    images_b = [s for s in glob.glob(path_to_images + "/img_b*" ) if os.path.isfile(s)]
    images_b.sort(key=lambda s: os.path.getmtime(s), reverse=reverse)
    
    images_fake = [s for s in glob.glob(path_to_images + "/fake_b*" ) if os.path.isfile(s)]
    images_fake.sort(key=lambda s: os.path.getmtime(s), reverse=reverse)
    
    # Display the pictures in a grid
    for idx, (a, f, b) in enumerate(zip(images_a, images_fake, images_b)):
        
        if (idx==display_max_examples) and (idx !=0):
            #Enough images have been displayed
            break
        
        img_a = np.array(Image.open(a))
        img_fake = np.array(Image.open(f))
        img_b = np.array(Image.open(b))
        
        plt.figure(figsize=(7,7))
        ax = plt.subplot(131)
        ax.imshow(img_a)
        ax.set_title("Source")  
        ax = plt.subplot(132)
        ax.imshow(img_fake)
        ax.set_title("Generated") 
        ax = plt.subplot(133)
        ax.imshow(img_b)
        ax.set_title("Target") 
          
        plt.show()
        plt.close()   
           
 
        
def display_dataset_examples(path_to_tfrecords, display_max_examples=3):
    ''' 
    Display a few image examples from the TFRecord dataset file
    of the image_a and image_b pair.
    
    Args:
        path_to_tfrecords (str) :  path to tfrecords file
        display_max_examples (int) : Number of samples to display                                   
    '''        
         
    #-----------------------------------------------
    #  display_dataset_examples - Helper Functions
    #-----------------------------------------------       
    def display_example(string_record):
        """
        Display the input and target images encoded in an example 
        read from the TFRecord file. 
        
        Args:
            string_record : string containing the serialized features 
            from the dataset
    
        """
        
        # Decode an example from the TFRecord
        example = tf.train.Example()
        example.ParseFromString(string_record)
                
        jpeg_file = (example.features.feature['jpeg_file']
                     .bytes_list                    
                     .value[0])

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])
        
        depth = int(example.features.feature['depth']
                    .int64_list
                    .value[0])

        raw_img_a = (example.features.feature['raw_img_a']
                 .bytes_list
                 .value[0])
        
        raw_img_b = (example.features.feature['raw_img_b']
                 .bytes_list
                 .value[0])

        # Reconstruct ths Image Numpy arrays (for the Input and Target images pair)       
        img_a_1d = np.fromstring(raw_img_a, dtype=np.uint8)
        img_a = img_a_1d.reshape((height, width, depth))
        img_b_1d = np.fromstring(raw_img_b, dtype=np.uint8)
        img_b = img_b_1d.reshape((height, width, depth))
        
   
        # Display the image
        print("Displaying {} (size {}x{})".format(jpeg_file.decode("utf-8"), width, height))
        plt.figure(figsize=(7,7))
        ax = plt.subplot(121)
        ax.imshow(img_a)
        ax.set_title("img_a")
        ax = plt.subplot(122)
        ax.imshow(img_b)
        ax.set_title("img_b")        
        plt.show()
        plt.close()
    
    
    #-----------------------------------------------
    #  display_dataset_examples - Main
    #-----------------------------------------------
    
    # Mute tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Define a tensorflow Input Pipeline (nothing to do with Kubeflow Pipelines)    
    record_iterator = tf.python_io.tf_record_iterator(path=path_to_tfrecords)
       
    steps=0 # Counter to track the nb of images in the dataset
    
    for string_record in record_iterator:

        # Decode and Display the Dataset images 
        if steps < display_max_examples:
            display_example(string_record)
            
        steps=steps+1
     
    print("{} contains {} images".format(path_to_tfrecords, steps-1))
    
    
    

