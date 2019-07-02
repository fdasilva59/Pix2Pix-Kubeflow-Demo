def download_dataset(fname:str , origin:str, cachedir:str="./", 
                     cachesubdir:str='datasets')-> str:
#
#------------- Change for KubeFlow Pipelines : Specify Types  ----
#  Use above function definition instead of: 
#
#      def download_dataset(fname, origin, cache_dir="./", 
#                           cache_subdir='datasets'):
#
#  ==> The function still works locally the same way
#-----------------------------------------------------------------
    """
    Download Pix2Pix datasets.
    
    By default the file at the url `origin` is downloaded to the
    cache_dir `./`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `./datasets/example.txt`.
    
    Args:
        fname (string) : file name of the dataset archive (i.e "maps.tar.gz")
        url (string)   : full url to download the dataset archive
        cache_dir (string)   
        cache_subdir (string)

    Returns:
        data_path (string) : path to uncompressed dataset directory
    """
    
    #------ Change for KubeFlow Pipelines : include Imports ---------
    #  As this function will be converted into a container operation
    #  we need to include its imports inside that function
    import tensorflow as tf
    import os
    #from pathlib import Path # TODO Remove
    #----------------------------------------------------------------
    
    try:
        # Use Keras.utils to download the dataset archive
        data_path = tf.keras.utils.get_file(fname, origin, 
                                            extract=True, 
                                            archive_format='auto', 
                                            cache_dir=cachedir, 
                                            cache_subdir=cachesubdir)

        data_path = os.path.join(os.path.dirname(data_path), 'maps/')
        print("Path location to the dataset images is {}".format(data_path))
        print("{} contains {}".format(data_path, os.listdir(data_path)))
        return data_path
    
    except:
        print('Failed to download the dataset at url {}'.format(URL))
        return None
    