def train_pix2pix(pathtfrecords:str, pathtflogs:str, pathoutputs:str, pathcheckpoints:str,
                  epochs:int=1, initialresize:int=286, cropresize:int=256, resizemethod:int=1,
                  saveevery:int=100, earlystop:int=0)->str:

    """ Build and Train Pix2Pix, a Conditional GAN neural Network for image translation.    

        Args:
            pathtfrecords (str)   : Full path to the TFRecords file containing the training dataset
            pathtflogs (str)      : Full path to the Tensorboard log directory
            pathoutputs (str)     : Full path to the generated images output directory
            pathcheckpoints (str) : Full path to the Tensorflow model checkpoints directory
            epochs (int)          : Nb of epochs for training
            initialresize (int)   : Target size for rezising input images
            cropresize (int)      : Target size for cropping the resized images
            resizemethod (int)    : Image resize method (BILINEAR=0, NEAREST_NEIGHBOR=1, BICUBIC=2, AREA=3)
            saveevery (int)       : Steps frequency to save logs, models, output images 
            earlystop (int)       : Stop training after N steps 
                      
        Returns:
            String path to the generated images output
    """
    

    # Need to install matplotlib inside the container where the code will be executed
    try:
        import matplotlib.pyplot as plt

    except ImportError as e:
        print("[INFO] Installing matplotlib")
        from pip._internal import main as pip
        # packages to install
        packages_to_install = ["matplotlib"]

        for package in packages_to_install:
            pip(["install",  "--disable-pip-version-check", "--upgrade", "--quiet", package])

    finally:
        import matplotlib.pyplot as plt


    import numpy as np
    import tensorflow as tf
    import json
    import time
    import os



    #------------------------------------------------
    #  Helper function to resize and perform
    #  data augmentation before feeding the input
    #  of the neural networks
    #------------------------------------------------
    def transform_image(img_a, img_b, initial_resize=286, crop_resize=256,
                        resize_method=resizemethod):
        """
        Resize a pair of  input/target Tensor Images to meet the
        Neural Network input size. It can also apply Data augmentation
        by by randomly resizing and cropping the pair of
        input/target Tensor images

        If crop_resize = 0 , then only resizing is applied
        If initial_resize > crop_resize, then Data augmentation is applied

        Args:
            img_a : 3D tf.float32 Tensor image
            img_b : 3D tf.float32 Tensor image
            initial_resize (int) : Initial resizing before Data augmentation
            crop_resize (int) : Final image size after random cropping
            resize_method (int) : BILINEAR=0, NEAREST_NEIGHBOR=1, BICUBIC=2, AREA=3

        Returns:
            img_a, img_b : Normalized 4D tf.float32 Tensors [ 1, height, width, channel ]
                           (with width = height = crop_resize)
        """

        #..........................................
        #  Transform_image nested helper functions
        #..........................................
        def _normalize(image):
            """ Normalize a 3D Tensor image """
            return (image / 127.5) - 1


        def _resize(img_a, img_b, size, resize_method):
            """ Resize a pair of Input/Target 3D Tensor Images """
            img_a = tf.image.resize_images(img_a, [size, size], method=resize_method)
            img_b = tf.image.resize_images(img_b, [size, size], method=resize_method)
            return img_a, img_b


        def _dataset_augment(img_a, img_b, initial_resize, crop_resize, resize_method):
            """ Dataset augmentation (random resize+crop) of a pair of source/target 3D Tensor images """

            # Resize image to size [initial_resize,  initial_resize, 3 ]
            img_a, img_b = _resize(img_a, img_b, initial_resize, resize_method)

            # Random cropping to size [crop_resize,  crop_resize, 3 ]
            # Images are staked to preserve source/target relationship
            stacked_images = tf.stack([img_a, img_b], axis=0) # size [2, h, w, ch]
            cropped_images = tf.image.random_crop(stacked_images,
                                                  size=[2, crop_resize, crop_resize, 3])
            img_a, img_b = cropped_images[0], cropped_images[1]

            # Random mirorring of the images
            if np.random.rand() > 0.5:
                img_a = tf.image.flip_left_right(img_a)
                img_b = tf.image.flip_left_right(img_b)

            return img_a, img_b

        #..........................................
        #  Transform_image Main
        #..........................................

        with tf.variable_scope('Transform'):

            # Normalize the Image Tensors
            img_a, img_b = _normalize(img_a), _normalize(img_b) # [ height, width, channel]

            # Check if data augmentation can be used
            if (initial_resize > crop_resize) and (crop_resize > 0):
                # Aply data augmenation
                img_a, img_b = _dataset_augment(img_a, img_b, initial_resize, crop_resize, resize_method)
            else:
                # Just Resize image to size [initial_resize,  initial_resize, 3 ]
                img_a, img_b = _resize(img_a, img_b, initial_resize, resize_method)

            # Add a batch dimension to have 4D Tensor images
            img_a = tf.expand_dims(img_a , axis=0)  # [ 1, height, width, channel]
            img_b = tf.expand_dims(img_b , axis=0)  # [ 1, height, width, channel]

        return img_a, img_b


    #------------------------------------
    #  Helper function to Save a Numpy
    #  image array to disk
    #------------------------------------
    def save_image(img, image_name, decode):
        if decode:
            # If necessary, drop the batch dimension
            if len(img.shape)== 4:
                img = np.squeeze(img, axis=0 )
            # Reverse the normalization process
            img = (img * 127.5) + 127.5
            img = img.astype(int).clip(0,255)

        # Write image to disk
        plt.imsave(image_name, img)



    #------------------------
    #  Helper functions to
    #  Build the Pix2Pix GAN
    #------------------------
    def compute_unet_specs(img_size=256, nb_channels=3, nb_filters=64, verbose=False):
        '''
        Compute the specs for the u-net networks to down scale, by a factor of 2,
        the image size down to 1, and then, to up scale , by a factor of 2,
        the image size up to the original image size

        Args:
            img_size: size of the input image (It is supposed that width = height)
            nb_channels: nb_channels in the input image (3 by default)
            nb_filters: number of initial filters after the u-net input

        Returns:
            encoder_filter_specs:  list of [filter_height, filter_width, in_channels, out_channels]
                                   used by tf.conv2d (down sampling)
            decoder_filter_specs:  list of [filter_height, filter_width, out_channels, oin_channels]
                                   used by tf.conv2d_transpose (up sampling)
            decoder_output_specs:  list of [batch_size, img_width, img_height, img_channels]
                                   used by tf.conv2d_transpose (up sampling)

        '''

        # Convolutions parameters
        kernel_size =4
        stride_size =2

        # Compute nb of layers needed to down scale,
        # by a factor of 2, the image size down to 1
        nb_layers = int(np.floor(np.log(img_size)/np.log(2)))

        # Nb of filters
        encoder_filters_list = [nb_channels]+ [nb_filters * min(8, (2**i)) for i in range(nb_layers)]
        decoder_filters_list = [nb_filters * min(8, (2**nb_layers))] + encoder_filters_list[:-1][::-1]
        if verbose:
            print("[INFO][Compute u-net specs] nb_layers", nb_layers)
            print("[INFO][Compute u-net specs] encoder_filters_list", encoder_filters_list)
            print("[INFO][Compute u-net specs] decoder_filters_list", decoder_filters_list)

        # Compute Encoder conv2d Filter Specs
        encoder_filter_specs = []
        for idx in range(nb_layers):
            kernel_specs = [kernel_size, kernel_size, encoder_filters_list[idx], encoder_filters_list[idx+1]]
            encoder_filter_specs.append(kernel_specs)
        if verbose:
            print("\n\n[INFO][Compute u-net specs] encoder_filter_specs {}".format(encoder_filter_specs))

        # Compute Decoder conv2d_transpose Filter specs
        decoder_filter_specs = []
        for idx in range(nb_layers):
            if idx==0:
                # the inner layers of the u-net networks dont need an extra residual skip connection
                kernel_specs = [kernel_size, kernel_size, decoder_filters_list[idx+1], decoder_filters_list[idx]]
            else:
                # In other layers, Input_channel size is x2 to match with the residual layer concatenation
                kernel_specs = [kernel_size, kernel_size, decoder_filters_list[idx+1], decoder_filters_list[idx]*2]
            decoder_filter_specs.append(kernel_specs)
        if verbose:
            print("\n\n[INFO][Compute u-net specs] decoder_filter_specs {}".format(decoder_filter_specs))

        # Compute Decoder Output Channel specs
        decoder_output_specs = []
        upsampling_size_list = [1 * (2**i) for i in range(1, len(decoder_filter_specs)+1)]
        if verbose:
            print("\n[INFO][Compute u-net specs] upsampling_size_list", upsampling_size_list)

        for idx in range(nb_layers):
            out_spec = [1,                            # Batch size is 1
                        upsampling_size_list[idx],    # Width
                        upsampling_size_list[idx],    # Height
                        decoder_filter_specs[idx][2]] # Channels
            out_shape = np.array(out_spec, dtype=np.int32 )
            decoder_output_specs.append(out_shape)
        if verbose:
            print("\n\n[INFO][Compute u-net specs] decoder_output_specs {}\n\n".format(decoder_output_specs))

        return encoder_filter_specs, decoder_filter_specs, decoder_output_specs



    def generator(generator_input_a, img_size=256, nb_channels=3, nb_filters=64, is_train=True, verbose=False):
        '''
        Build the Conditional GAN Generator using a U-net architecture

        Args:
            generator_input_a (Tensor) : Input image 4D Tensor [1, img_size, img_size, nb_channels]
            img_size (int)    : size of the input image (It is supposed that width = height)
            nb_channels (int) : nb_channels in the input image (3 by default)
            nb_filters (int)  : number of initial filters for the first convolution
            is_train (Bool)   : True if training mode
            verbose (Bool)    : Verbose logs

        Returns:
            output_generator (Tensor) : Output image 4D Tensor [1, img_size, img_size, nb_channels]
        '''

        LEAKY_RELU = 0.3  # 0.2 in paper 

        with tf.variable_scope('generator', reuse=False if is_train==True else True):

            # Compute the specifications for the u-net network
            encoder_filter_specs, decoder_filter_specs, decoder_output_specs = compute_unet_specs(img_size=256,
                                                                                                  nb_channels=3,
                                                                                                  nb_filters=64,
                                                                                                  verbose=verbose)
            ue = generator_input_a
            print("\n[INFO][Build u-net] unet_input_{}\n".format(generator_input_a.shape))


            #------------------------
            # Define u-net encoder
            #------------------------
            encoder_layers = []
            for idx in range(len(encoder_filter_specs)):
                with tf.variable_scope("unet_enc_%s_block" % idx):


                    # ----- Add LeakyReLU except for first layer
                    if idx>0:
                        ue = tf.nn.leaky_relu(features=ue,
                                              alpha=LEAKY_RELU,
                                              name="unet_enc_%s_leakyrelu" % idx)
                        if verbose:
                            print("   [DEBUG][Build u-net encoder] unet_dec_{}_leakyrelu shape={}".format(idx, ue.shape))

                    # ----- Add Conv2D with u-net specs
                    kernel = tf.Variable(tf.random_normal(shape=encoder_filter_specs[idx],
                                                          mean=0.0, stddev=0.02))
                    ue = tf.nn.conv2d(input=ue,
                                     filter=kernel,
                                     strides=[1,2,2,1],
                                     padding='SAME',
                                     name="unet_enc_%s_conv" % idx)
                    if verbose:
                            print("   [DEBUG][Build u-net encoder] unet_dec_{}_conv2d shape={}".format(idx, ue.shape))

                    # ----- Add Batch Normalization except for first and last layer
                    if idx >0 and idx<len(encoder_filter_specs)-1:
                        ###### TODO change to tf.nn.batch_normalization(...  )
                        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
                        ue = tf.layers.batch_normalization(inputs = ue,
                                                           axis=3,
                                                           epsilon=1e-3, #1e-5,
                                                           momentum=0.99, #0.1,
                                                           training=True,
                                                           gamma_initializer=initializer,
                                                           name="unet_enc_%s_batchnorn" % idx)

                        if verbose:
                            print("   [DEBUG][Build u-net encoder] unet_dec_{}_BatchNorm shape={}".format(idx, ue.shape))


                    # Keep track of the encoder layers to later add the residual connections
                    encoder_layers.append(ue)

                    print("[INFO][Build u-net encoder] unet_enc_{} layer output shape={}".format(idx, ue.shape))
                    if verbose:
                        print("\n")


            #------------------------
            # Define u-net decoder
            #------------------------

            ud = ue # U-net decoder input = U-net encoder output

            # U-net decoder residual connections are added in reverse order of u-net encoder layers
            encoder_residuals = encoder_layers[::-1]

            for idx in range(len(decoder_filter_specs)):
                with tf.variable_scope("unet_dec_%s_block" % idx):

                    # ----- Add ReLU
                    ud = tf.nn.relu(features=ud,
                                    name="unet_dec_%s_relu" % idx)
                    if verbose:
                        print("   [DEBUG][Build u-net decoder] unet_dec_{}_relu shape={}".format(idx, ud.shape))


                    # ----- Add transposed 2D convolution Layer
                    kernel = tf.Variable(tf.random_normal(shape=decoder_filter_specs[idx],
                                                          mean=0.0, stddev=0.02))  ### TODO specify other mean/stddev ?)
                    out_size = decoder_output_specs[idx]
                    if verbose:
                        print("   [DEBUG][Build u-net decoder] filter_specs {}".format(decoder_filter_specs[idx]))
                        print("   [DEBUG][Build u-net decoder] output_shape {}".format(out_size))

                    ud = tf.nn.conv2d_transpose(value=ud,
                                                filter=kernel,
                                                output_shape=out_size,
                                                strides=[1,2,2,1],
                                                padding='SAME',
                                                name= "unet_dec_%s_conv" % idx)
                    if verbose:
                        print("   [DEBUG][Build u-net decoder] unet_dec_{}_deconv shape={}".format(idx, ud.shape))


                    # ----- Add Batch Normalization except for last layer
                    if idx<len(decoder_filter_specs)-1:
                        ##### TODO change to tf.nn.batch_normalization(...  )
                        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
                        ud = tf.layers.batch_normalization(inputs = ud,
                                                           axis=3,
                                                           epsilon=1e-3, #1e-5,
                                                           momentum=0.99, #0.1,
                                                           training=True,
                                                           gamma_initializer=initializer,
                                                           name="unet_dec_%s_batchnorn" % idx)

                        if verbose:
                                print("   [DEBUG][Build u-net decoder] unet_dec_{}_batchnorm shape={}".format(idx, ud.shape))

                    # ----- Add Dropout only for the 3 first layers
                    if idx <3:
                        ud = tf.nn.dropout(x=ud,
                                           keep_prob=0.5,
                                           name="unet_dec_%s_dropout" % idx)
                        if verbose:
                                print("   [DEBUG][Build u-net decoder] unet_dec_{}_dropout shape={}".format(idx, ud.shape))


                    # ----- Add residual connection to u-net encoder and decoder (except for the outtermost layers)
                    if idx < len(decoder_filter_specs)-1:
                        if verbose:
                            print("   [DEBUG][Build u-net decoder] add residual concat ud.shape={} encoder_residuals[idx+1].shape={}".format(ud.shape, encoder_residuals[idx+1].shape))
                        ud = tf.concat([ud, encoder_residuals[idx+1]], axis=-1, name="unet_dec_%s_res" % idx)


                    # ----- Add final activation at the output of the u-net decoder
                    if idx == len(decoder_filter_specs)-1:
                        ud = tf.tanh(x=ud,
                                     name="unet_dec_%s_tanh" % idx)
                        if verbose:
                            print("   [DEBUG][Build u-net decoder] add tanh final activation shape={}".format(ud.shape))


                    print("[INFO][Build u-net decoder] unet_dec_{} layer output shape={}".format(idx, ud.shape))
                    if verbose:
                        print("\n")

            output_generator = ud

            return output_generator



    def compute_pixelgan_specs(img_size=256, nb_channels=3, nb_filters=64, nb_layers=3, verbose=False):
        '''
        Compute the specs for the PixelGAN network (discriminator)

        Args:
            img_size: size of the input image (It is supposed that width = height)
            nb_channels: nb_channels in the input image (3 by default)
            nb_filters: number of initial filters for the first convolution
            nb_layers : nb of convolution layers (excluding input and output convolutions)

        Returns:
           filter_specs:  list of [filter_height, filter_width, in_channels, out_channels]
                                   used by tf.conv2d

        '''

        kernel_size = 4

        # Nb of filters (manually adding input and ouput channel/dimension size)
        filters_list = [nb_channels * 2 ] + [nb_filters * min(8, (2**i)) for i in range(nb_layers+1)]
        filters_list = filters_list + [ 1 ]
        print("\n[INFO][Compute PixelGAN specs] filters_list {}\n".format(filters_list))

        # Compute PixelGAN filter specs
        filter_specs = []
        for idx in range(nb_layers+2):
            kernel_specs = [kernel_size, kernel_size, filters_list[idx], filters_list[idx+1]]
            filter_specs.append(kernel_specs)
        if verbose:
            print("\n[INFO][Compute PixelGAN specs] filter_specs {}\n".format(filter_specs))

        return filter_specs



    def discriminator(discriminator_input_a, discriminator_input_b, img_size=256, nb_channels=3,
                      nb_filters=64, nb_layers= 3, is_train=True, reuse = False, verbose=False):
        '''
        Build the "PatchGAN" Discriminator (Conditional GAN). In Pix2PIx, this is a "70x70 PatchGAN" 
        
        (In Pix2Pix it returns an output Tensor of shape [1, 30, 30, 1] with a receptive fields of 70x70.
        See an online Field of Depth Calculator: 
        https://fomoro.com/research/article/receptive-field-calculator#4,2,1,SAME;4,2,1,SAME;4,2,1,SAME;4,1,1,SAME;4,1,1,SAME)

        Args:
            discriminator_input_a (Tensor) : Input image 4D Tensor [1, img_size, img_size, nb_channels]
            discriminator_input_b (Tensor) : Input image 4D Tensor [1, img_size, img_size, nb_channels]
            img_size (int)    : size of the input image (It is supposed that width = height)
            nb_channels (int) : nb_channels in the input image (3 by default)
            nb_filters (int)  : number of initial filters for the first convolution
            nb_layers (int)   : nb of convolution layers (excluding input and output convolutions)
            is_train (Bool)   : True if training mode
            reuse  (Bool)     : Reuse the variable for the GAN discriminator
            verbose (Bool)    : Verbose logs

        Returns:
            output_discriminator (Tensor) : Output image 4D Tensor [1, N, N, 1]
        '''

        with tf.variable_scope("discriminator",reuse=reuse):

            LEAKY_RELU = 0.2

            # Compute the specs for the conv2d layers
            filter_specs = compute_pixelgan_specs(img_size=img_size,
                                                  nb_channels=nb_channels,
                                                  nb_filters=nb_filters,
                                                  nb_layers=nb_layers,
                                                  verbose=verbose)

            sride_val = 2  # Default srides value for the convolutions
            pad_val='SAME' # Default padding value for the convolutions


            # ---------------------------------------
            # Define the Input of the discriminator
            # ---------------------------------------
            discriminator_input = tf.concat([discriminator_input_a, discriminator_input_b],
                                            axis=-1, name="discriminator_input")

            d = discriminator_input

            for idx in range(len(filter_specs)):

                with tf.variable_scope("discriminator_%s_block" % idx):


                    # ----- Add transposed 2D convolution Layer

                    # Add zero padding for the last 2 convolutions layers
                    if idx >= (len(filter_specs) - 2):
                        d = tf.pad(tensor=d,
                                   paddings=[[0,0],[1,1], [1,1],[0,0]],
                                   mode="CONSTANT", ## NEW
                                   name="discriminator_%s_zeropad" % idx)

                        if verbose:
                            print("   [DEBUG][Build pixelgan] discrimininator_{}_zeropad shape={}".format(idx, d.shape))

                    kernel = tf.Variable(tf.random_normal(shape=filter_specs[idx],
                                                          mean=0.0, stddev=0.02))
                    if verbose:
                        print("   [DEBUG][Build pixelgan] kernel.shape =", kernel.shape)
                        print("   [DEBUG][Build pixelgan] filter_specs {}".format(filter_specs[idx]))


                    # Use strides of 2 for the convolution except for the last 2 convolutions
                    if idx < len(filter_specs)-2:
                        stride_val = 2
                        pad_val='SAME'
                    else:
                        stride_val = 1
                        pad_val='VALID'

                    # Add the convolution layer
                    d = tf.nn.conv2d(input=d,
                                     filter=kernel,
                                     strides=[1, stride_val, stride_val, 1],
                                     padding=pad_val,
                                     name="discriminator_%s_conv" % idx)
                    if verbose:
                        print("   [DEBUG][Build pixelgan] discrimininator_{}_deconv shape={} (stride {}\{})".format(idx, d.shape, stride_val, stride_val))


                    # ----- Add Batch Normalization except for first and last layers
                    if idx >0 and idx<len(filter_specs)-1:
                        ###### TODO change to tf.nn.batch_normalization(...  )
                        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
                        d = tf.layers.batch_normalization(inputs = d,
                                                          axis=3,
                                                          epsilon=1e-3, #1e-5,
                                                          momentum=0.99, #0.1,
                                                          training=is_train,
                                                          gamma_initializer=initializer,
                                                          name="discrimininator_%s_batchnorn" % idx)
                        if verbose:
                            print("   [DEBUG][Build pixelgan] discrimininator_{}_batchnorm shape={}".format(idx, d.shape))

                    # ----- Add LeakyReLU except for last layer
                    if idx<len(filter_specs)-1:
                        d = tf.nn.leaky_relu(features=d,
                                              alpha=LEAKY_RELU,
                                              name="discrimininator__%s_leakyrelu" % idx)
                        if verbose:
                            print("   [DEBUG][Build pixelgan] discrimininator_{}_leakyrelu shape={}".format(idx, d.shape))

                    print("[INFO][Build pixelgan] discrimininator__{} layer output shape={}".format(idx, d.shape))
                    if verbose:
                        print("\n")

            return d

    def model_loss(img_a, img_b, loss_lambda=100):

        EPS=1e-12



        # ----------------------------------------------
        # Compute Generator and Discriminator outputs
        # ----------------------------------------------
        fake_b = generator(generator_input_a=img_a,
                           is_train=True, verbose=False)

        d_model_real = discriminator(discriminator_input_a=img_a,
                                     discriminator_input_b=img_b,
                                     is_train=True, reuse=False, verbose=False)

        d_model_fake = discriminator(discriminator_input_a=img_a,
                                     discriminator_input_b=fake_b,
                                     is_train=True, reuse=True, verbose=False)

        # ------------------------------
        # Compute Discriminator loss
        # ------------------------------

        with tf.variable_scope("Discriminator_Loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_model_real),
                                                                                 logits=d_model_real))

            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_model_fake),
                                                                                 logits=d_model_fake))
            d_loss = d_loss_real + d_loss_fake

            ## Alternative version using tf.keras API
            #loss_object = tf.keras.losses.binary_crossentropy(from_logits=True)
            #d_loss_real = loss_object(y_true=tf.ones_like(d_model_real), y_pred=d_model_real)
            #d_loss_fake = loss_object(y_true=tf.zeros_like(d_model_fake), y_pred=d_model_fake)
            #d_loss = d_loss_real + d_loss_fake

        # ------------------------------
        # Compute Generator loss
        # ------------------------------
        with tf.variable_scope("Generator_Loss"):
            with tf.variable_scope("Generator_GAN_Loss"):
                g_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_model_fake),
                                                                                    logits=d_model_fake))
                ## Alternative version using tf.keras API
                #g_loss_gan = loss_object(y_true=tf.ones_like(d_model_fake), y_pred=d_model_fake)


            with tf.variable_scope("Generator_L1_Loss"):
                g_loss_l1 = tf.reduce_mean(tf.abs(fake_b - img_b))

            g_loss = g_loss_gan + loss_lambda * g_loss_l1

        # -----------------------------------
        # Collect Summaries for Tensorboard
        # -----------------------------------
        tf.summary.scalar('Discriminator Loss', d_loss)
        tf.summary.scalar('Generator Loss', g_loss)
        tf.summary.scalar('Generator GAN Loss', g_loss_gan)
        tf.summary.scalar('Generator L1 Loss', g_loss_l1)
        tf.summary.image("fake_b", fake_b, max_outputs=1)

        return d_loss, g_loss, g_loss_gan, g_loss_l1, fake_b, d_model_real, d_model_fake


    def model_opt(g_loss, d_loss, lr=2e-4, beta_1=0.5):

        # Get weights and bias to updates
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

        # Optimize
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            # Bugfix: avoid noise artifact to appears in generated images 
            # Instead of using tf.train.AdamOptimizer(...).minimize(...) 
            # directly, we apply Gradient Clipping
            g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1)
            gradients = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            g_train_opt = g_optimizer.apply_gradients(clipped_gradients)


            # Bugfix: avoid noise artifact to appears in generated images 
            # Instead of using tf.train.AdamOptimizer(...).minimize(...) 
            # directly, we apply Gradient Clipping
            d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1)
            gradients = d_optimizer.compute_gradients(d_loss, var_list=d_vars)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            d_train_opt = d_optimizer.apply_gradients(clipped_gradients)

        return g_train_opt, d_train_opt


    ###################################################
    #                      Main
    #                 Training Loop
    ###################################################

    # Mute tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Reset the graph (in case of multiple execution in Jupyter Notebooks)
    tf.reset_default_graph()

    # Parameters for tf.data
    buffersize = 400
    parallel = 4

    # Use named subdirs for each training runs
    timestamp =  time.strftime("%d-%H%M%S")
    subdir ="/run-" + timestamp
    path_to_tfrecords = pathtfrecords  # file path name
    path_to_tflogs = pathtflogs + subdir
    path_to_outputs = pathoutputs + subdir
    path_to_checkpoints = pathcheckpoints + subdir
    os.makedirs(path_to_tflogs)
    os.makedirs(path_to_outputs)


    # ------------------------------
    # Create a Dataset iterator
    # for the training Dataset
    # ------------------------------

    # Create a Dataset from the TFRecord file
    raw_image_dataset = tf.data.TFRecordDataset(path_to_tfrecords)

    # Parse each example to extract the features
    feature = { 'jpeg_file' : tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'raw_img_a': tf.FixedLenFeature([], tf.string),
                'raw_img_b': tf.FixedLenFeature([], tf.string)
    }

    def _parse_image_function(example_proto):
        return tf.parse_single_example(example_proto, feature)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset = parsed_image_dataset.repeat(epochs)
    parsed_image_dataset = parsed_image_dataset.shuffle(buffersize)

    # Create an iterator on the Dataset
    iterator = parsed_image_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    ## TODO : Define as hyperparameters
    source_size = 600
    img_size = 256
    nb_channels = 3
    batch_size=1


    # ------------------------------
    # Build the model
    # ------------------------------

    # Placeholders for input 3D Tensor images
    source_image_a = tf.placeholder(dtype=tf.float32, shape=(source_size, source_size, nb_channels), name="image_a")
    source_image_b = tf.placeholder(dtype=tf.float32, shape=(source_size, source_size, nb_channels), name="image_b")


    image_a, image_b = transform_image(source_image_a, source_image_b,
                                       initialresize, cropresize, resize_method=resizemethod)
    # Collect summaries to display in Tensorboard
    tf.summary.image("image_a", image_a, max_outputs=1)
    tf.summary.image("image_b", image_b, max_outputs=1)

    # Cunters to track the training progression
    steps=0

    # Compute the Loss (and retrieve the fake_b image created by the generator)
    d_loss, g_loss, g_loss_gan, g_loss_l1, fake_b, d_model_real, d_model_fake \
         = model_loss(img_a=image_a,img_b=image_b,loss_lambda=100)


    # Optimize
    g_train_opt, d_train_opt = model_opt(g_loss=g_loss,
                                         d_loss=d_loss,
                                         lr=2e-4, beta_1=0.5)


    # ------------------------------
    # Train the model
    # ------------------------------

    try:
        print("\n\nStart Pix2Pix Training")
        start_time = time.time()

        with tf.Session() as sess:

            ## Uncomment if Calling Keras layers on TensorFlow tensors
            #from keras import backend as K
            #K.set_session(sess)

            # Enable Kubeflow Pipelines UI built-in support for Tensorboard
            try:
                metadata = {
                    'outputs' : [{
                        'type': 'tensorboard',
                        'source': pathtflogs,
                    }]
                }

                # This works only inside Docker containers
                with open('/mlpipeline-ui-metadata.json', 'w') as f:
                    json.dump(metadata, f)

            except PermissionError:
                pass

            # Save the Session graph for Tensorboard
            train_writer = tf.summary.FileWriter(path_to_tflogs + '/train', sess.graph)

            sess.run(tf.global_variables_initializer())

            steps=1

            # Loop over the training Dataset
            while True:
                try:

                    # get one example from teh TFRecord file
                    image_features  = sess.run(next_element)

                    # Extract the individual features and  reconstruct
                    # the input images to feed the Neural Networks
                    width = int(image_features['width'])
                    height = int(image_features['height'])
                    depth = int(image_features['depth'])
                    raw_img_a = image_features['raw_img_a']
                    a_image = np.frombuffer(raw_img_a, dtype=np.uint8)
                    a_image = a_image.reshape((height,width,depth))
                    raw_img_b = image_features['raw_img_b']
                    b_image = np.frombuffer(raw_img_b, dtype=np.uint8)
                    b_image = b_image.reshape((height,width,depth))


                    # Train the discriminator
                    _ = sess.run(d_train_opt, feed_dict={source_image_a: a_image,
                                                         source_image_b: b_image})

                    # Train the generator
                    _ = sess.run(g_train_opt, feed_dict={source_image_a: a_image,
                                                         source_image_b: b_image})

                    # From time to time, display statistics, collect logs, save images to disk
                    if (steps % saveevery == 0) or (steps==1) or (steps==earlystop):

                        # Merge all the summaries for Tensorboard
                        merged = tf.summary.merge_all()

                        # Evaluate all the summaries, metrics and Images to collect
                        summary, train_d_loss, train_g_loss, train_g_loss_gan, train_g_loss_l1, fb, a, b \
                            = sess.run([merged, d_loss, g_loss, g_loss_gan, g_loss_l1, fake_b, image_a, image_b ],
                                     feed_dict={source_image_a: a_image, source_image_b: b_image})

                        # Write summaries
                        train_writer.add_summary(summary, steps)

                        # TODO : decide if we keep this (=> Images are also captured by Tensorboard)
                        # Save Pix2Pix images to disk (use timestamp to have unique file names)
                        timestamp =  time.strftime("%d-%H%M%S")
                        img_name = path_to_outputs + "/img_a-" + str(timestamp) + ".png"
                        save_image(a, img_name, decode=True)
                        img_name = path_to_outputs + "/img_b-" + str(timestamp) + ".png"
                        save_image(b, img_name, decode=True)
                        img_name = path_to_outputs + "/fake_b-" + str(timestamp) + ".png"
                        save_image(fb, img_name, decode=True)

                        # Print Monitoring info on stdout
                        print("D-Loss:{:.5f}...".format(train_d_loss),
                                  "G-Loss_TOTAL:{:.5f}".format(train_g_loss),
                                  "G-loss_L1:{:.4f}".format(train_g_loss_l1),
                                  "G-loss_GAN:{:.4f}".format(train_g_loss_gan),
                                  "Elapsed_time ={:.1f} min".format((time.time()-start_time)/60),
                                  "Steps={}".format(steps))

                        # Save checkpoints
                        path_to_chk = path_to_checkpoints + "/chk-" + time.strftime("%d-%H%M%S")
                        tf.saved_model.simple_save(session=sess, export_dir=path_to_chk,
                                                   inputs={"source_image_a": source_image_a,
                                                           "source_image_b": source_image_b},
                                                   outputs={"fake_b": fake_b,
                                                            "d_model_real": d_model_real,
                                                            "d_model_fake": d_model_fake,
                                                    })


                    if (steps==earlystop):
                        # Early stop at max steps
                        print("Early stop at steps {}".format(steps))
                        break
                    steps +=1


                except tf.errors.OutOfRangeError:
                    print("End of dataset")
                    # Save final checkpoints before exiting training loop
                    path_to_chk = path_to_checkpoints + "/chk-" + time.strftime("%d-%H%M%S")
                    tf.saved_model.simple_save(session=sess, export_dir=path_to_chk,
                                               inputs={"source_image_a": source_image_a,
                                                       "source_image_b": source_image_b},
                                               outputs={"fake_b": fake_b,
                                                        "d_model_real": d_model_real,
                                                        "d_model_fake": d_model_fake,
                                                })
                    break

        print("*** End of Training ***")
        return path_to_outputs

    except KeyboardInterrupt:
        # Catch error when aborting the  training in Jupyter Notebook
        print("Training aborted")
        pass
    return path_to_outputs


