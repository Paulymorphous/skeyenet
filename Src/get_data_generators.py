"""
Filename: get_data_generators.py

Function: Builds and returns data generators.

Author: Jerin Paul (https://github.com/Paulymorphous)
Website: https://www.livetheaiexperience.com/
"""

from keras.preprocessing.image import ImageDataGenerator

def GetDataGenerators(augmentation_parameters, train_images_path=None, train_targets_path=None, test_images_path=None, test_targets_path=None, batch_size = 64, seed=42):
    """
        Builds and returns ImageDataGenerators based on the paths that are sepcified. 

        Please note:
        > Since this is not a multi-class classification problem and ImageDataGenerators require atleast one folder with images in it, we provide the path to the parent directory of the folder containting the images.
        > If you want only one type of data generator, then you need to provide the path only for that dataset. See Example for clarification.
        > A Validation generator is also returned with the Train generator.
        > You can modify the parameters for Image Augmentation in the Config File.

        Read more about the prerocessing methods here: https://keras.io/api/preprocessing/image/

        Parameters
        ----------
        >augmentation_parameters (Object of ConfigParser SectionProxy): Configurations from for Augmentation.
        >train_images_path (str): Path to the parent folder of the folder containting Train Images.
        >train_targets_path (str): Path to the parent folder of the folder containting Train Masks.
        >test_images_path (str): Path to the parent folder of the folder containting Test Images.
        >test_targets_path (str): Path to the parent folder of the folder containting Test Masks.
        >batch_size (int): Desired batch size for the datagenerators. Default: 64.
        >seed (int): this number seeds the Datagenerators. Default: 42, because its the answer to everything ;)

        Returns
        ----------
        > A list which can have some or all of the three types of Datagenerators - Train data generator, validation data generator, or/and Test data generator.

        Example
        ----------
        > If you want to test your model and only require the Test Datagenerator:
            test_generator = GetDataGenerators(test_images_path= <Path to test Images>,
                                            test_targets_path= <Path to test Maskss>,
                                            batch_size = 128
                                            )
    """

    generators = []

    def get_boolean(string):
        if string == "True":
            return True
        elif string == "False":
            return False
       
    if train_images_path and  train_targets_path:

        train_datagen = ImageDataGenerator(
                                            featurewise_center=get_boolean(augmentation_parameters["featurewise_center"]),
                                            samplewise_center=get_boolean(augmentation_parameters["samplewise_center"]),
                                            featurewise_std_normalization=get_boolean(augmentation_parameters["featurewise_std_normalization"]),
                                            samplewise_std_normalization=get_boolean(augmentation_parameters["samplewise_std_normalization"]),
                                            zca_whitening=get_boolean(augmentation_parameters["zca_whitening"]),
                                            zca_epsilon=float(augmentation_parameters["zca_epsilon"]),
                                            rotation_range=float(augmentation_parameters["rotation_range"]),
                                            width_shift_range=float(augmentation_parameters["width_shift_range"]),
                                            height_shift_range=float(augmentation_parameters["height_shift_range"]),
                                            shear_range=float(augmentation_parameters["shear_range"]),
                                            zoom_range=float(augmentation_parameters["zoom_range"]),
                                            channel_shift_range=float(augmentation_parameters["channel_shift_range"]),
                                            fill_mode=augmentation_parameters["fill_mode"],
                                            cval=float(augmentation_parameters["cval"]),
                                            horizontal_flip=get_boolean(augmentation_parameters["horizontal_flip"]),
                                            vertical_flip=get_boolean(augmentation_parameters["vertical_flip"]),
                                            rescale=float(augmentation_parameters["rescale"]),
                                            validation_split = float(augmentation_parameters["validation_split"])
                                           )

        train_image_generator = train_datagen.flow_from_directory(
                                                                 directory = train_images_path,
                                                                 batch_size = batch_size,
                                                                 class_mode=None,
                                                                 subset = 'training',
                                                                 seed = seed
                                                                )

        train_target_generator = train_datagen.flow_from_directory(
                                                                 directory = train_targets_path,
                                                                 batch_size = batch_size,
                                                                 class_mode=None,
                                                                 subset = 'training',
                                                                 seed = seed
                                                                )


        validation_image_generator = train_datagen.flow_from_directory( 
                                                                           directory = train_images_path,
                                                                           batch_size = batch_size,
                                                                           class_mode=None,
                                                                           subset = 'validation',
                                                                           seed = seed
                                                                      )

        validation_target_generator = train_datagen.flow_from_directory( 
                                                                           directory = train_targets_path,
                                                                           batch_size = batch_size,
                                                                           class_mode=None,
                                                                           subset = 'validation',
                                                                           seed = seed,
                                                                    )

        train_generator = zip(train_image_generator, train_target_generator)
        validation_generator = zip(validation_image_generator, validation_target_generator)
        
        generators.extend([train_generator, validation_generator])
    
    
    if test_images_path and  test_targets_path:
        
    
        test_datagen = ImageDataGeneratorImageDataGenerator(
                                            featurewise_center=get_boolean(augmentation_parameters["featurewise_center"]),
                                            samplewise_center=get_boolean(augmentation_parameters["samplewise_center"]),
                                            featurewise_std_normalization=get_boolean(augmentation_parameters["featurewise_std_normalization"]),
                                            samplewise_std_normalization=get_boolean(augmentation_parameters["samplewise_std_normalization"]),
                                            zca_whitening=get_boolean(augmentation_parameters["zca_whitening"]),
                                            zca_epsilon=float(augmentation_parameters["zca_epsilon"]),
                                            rotation_range=float(augmentation_parameters["rotation_range"]),
                                            width_shift_range=float(augmentation_parameters["width_shift_range"]),
                                            height_shift_range=float(augmentation_parameters["height_shift_range"]),
                                            shear_range=float(augmentation_parameters["shear_range"]),
                                            zoom_range=float(augmentation_parameters["zoom_range"]),
                                            channel_shift_range=float(augmentation_parameters["channel_shift_range"]),
                                            fill_mode=augmentation_parameters["fill_mode"],
                                            cval=float(augmentation_parameters["cval"]),
                                            horizontal_flip=get_boolean(augmentation_parameters["horizontal_flip"]),
                                            vertical_flip=get_boolean(augmentation_parameters["vertical_flip"]),
                                            rescale=float(augmentation_parameters["rescale"]),
                                            validation_split = float(augmentation_parameters["validation_split"])
                                           )


        test_image_generator = test_datagen.flow_from_directory(
                                                                  directory = test_images_path,
                                                                  batch_size = batch_size,
                                                                  class_mode=None,
                                                                  shuffle = False,
                                                                  seed = seed
                                                              )


        test_target_generator = test_datagen.flow_from_directory(
                                                                  directory = test_targets_path,
                                                                  batch_size = batch_size,
                                                                  class_mode=None,
                                                                  shuffle = False,
                                                                  seed = seed
                                                              )
        
        test_generator = zip(test_image_generator, test_target_generator)

        generators.append(test_generator)
    
    if generators:
        
        return generators
    
    else:
        print("Invalid Input for Data Paths. Plese Check and Retry.")
        return None