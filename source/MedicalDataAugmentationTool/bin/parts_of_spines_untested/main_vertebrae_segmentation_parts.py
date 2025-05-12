#!/usr/bin/python
import argparse
from collections import OrderedDict
from glob import glob

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from dataset import Dataset
from network import SpatialConfigurationNet, Unet
from tqdm import tqdm
from skimage import measure
import pandas as pd
import os
import joblib

import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.sitk_image
import utils.sitk_np
import utils.np_image
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler


class MainLoop(MainLoopBase):
    def __init__(self, config):
        """
        Initializer.
        :param cv: The cv fold. 0, 1, 2 for CV; 'train_all' for training on whole dataset.
        :param config: config dictionary
        """
        super().__init__()
        gpu_available = tf.test.gpu_device_name() != ''
        self.use_mixed_precision = gpu_available
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.cv = config.cv
        self.config = config
        self.batch_size = 1
        self.num_labels = 1
        self.num_labels_all = 27
        self.data_format = 'channels_last'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        self.network = Unet
        self.save_output_images = False
        self.save_debug_images = False
        self.image_folder = config.image_folder
        self.setup_folder = config.setup_folder
        self.output_folder = config.output_folder
        self.load_model_filenames = config.load_model_filenames
        self.image_size = [128, 128, 96]
        self.image_spacing = [config.spacing] * 3
        self.heatmap_size = self.image_size

        images_files = sorted(glob(os.path.join(self.image_folder, '*.nii.gz')))
        self.image_id_list = list(map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files))


        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.landmark_mapping_inverse = dict([(self.landmark_labels[i], i) for i in range(26)])

        self.train_distribution = np.array([[17.40229835, 2.3218056 ],
                                            [19.05028643, 1.53235943],
                                            [16.21117155, 1.75142291],
                                            [16.18960192, 1.69779387],
                                            [16.32604093, 1.51519982],
                                            [17.61978089, 1.54666049],
                                            [18.95499949, 1.4710998 ],
                                            [21.20228033, 1.74068245],
                                            [22.35513377, 1.78316327],
                                            [22.39719006, 1.6249268 ],
                                            [22.84898256, 1.61521746],
                                            [23.42713597, 1.51557237],
                                            [24.03896842, 1.70067131],
                                            [24.53292329, 1.66230595],
                                            [24.94033849, 1.69959017],
                                            [25.77855859, 1.70423575],
                                            [27.74191434, 1.86287431],
                                            [29.95497532, 1.99423831],
                                            [32.18330096, 2.18454973],
                                            [33.55364202, 2.24838495],
                                            [34.33177047, 2.65835002],
                                            [34.09609406, 2.81691356],
                                            [32.25845245, 2.96422197],
                                            [30.82805648, 2.37693112]])

        #if self.data_format == 'channels_first':
            #self.call_model = tf.function(self.call_model, input_signature=[tf.TensorSpec(tf.TensorShape([1, 2] + list(reversed(self.image_size))), tf.float16 if self.use_mixed_precision else tf.float32)])
        #else:
            #self.call_model = tf.function(self.call_model, input_signature=[tf.TensorSpec(tf.TensorShape([1] + list(reversed(self.image_size))) + [2], tf.float16 if self.use_mixed_precision else tf.float32)])

    def init_model(self):
        # create sigmas variable
        self.model = self.network(num_labels=self.num_labels, **self.network_parameters)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.output_folder, use_timestamp=False, files_to_copy=[])

    def init_datasets(self):
        self.valid_landmarks_file = os.path.join(self.setup_folder, 'vertebrae_localization/valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

        dataset_parameters = dict(image_base_folder=self.image_folder,
                                  setup_base_folder=self.setup_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  input_gaussian_sigma=0.75,
                                  heatmap_sigma=3.0,
                                  generate_single_vertebrae_heatmap=True,
                                  output_image_type=np.float16 if self.use_mixed_precision else np.float32,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()
        self.network_image_size = list(reversed(self.image_size))

    def call_model(self, image):
        return self.model(image, training=False)

    def test_full_image(self, dataset_entry):
        """
        Perform inference on a dataset_entry with the validation network.
        :param dataset_entry: A dataset entry from the dataset.
        :return: input image (np.array), network prediction (np.array), transformation (sitk.Transform)
        """
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        image = np.expand_dims(generators['image'], axis=0)
        single_heatmap = np.expand_dims(generators['single_heatmap'], axis=0)
        image_heatmap_concat = tf.concat([image, single_heatmap], axis=1 if self.data_format == 'channels_first' else -1)
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.load_model(load_model_filename)
            prediction = tf.sigmoid(self.call_model(image_heatmap_concat))
            predictions.append(prediction.numpy())
        prediction = np.mean(predictions, axis=0)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def getCenterpoints(self, segm):
        labels = np.unique(segm)[1:]
        c = np.zeros((len(labels), 4))
        for i, ver in enumerate(labels):
            if ver != 0:
                mask = (segm == ver) * 1
                props = measure.regionprops(mask)
                c[i] = np.append(np.round(np.array(props[0].centroid)).astype(int), int(ver))
        return c

    def getDistances(self, center_points):
        num_stdev = np.full(25, np.nan)
        labels = center_points[:,3].astype(int)
        cp = center_points[:, :3]
        for i, lab in enumerate(labels[:-1]):
            distance = np.linalg.norm(cp[i] - cp[i+1])
            num_stdev[lab-1] = (distance - self.train_distribution[lab-1, 0]) / self.train_distribution[lab-1, 1]
        return num_stdev

    def test(self, check):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        if len(self.load_model_filenames) == 1:
            self.load_model(self.load_model_filenames[0])

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        filter_largest_cc = True

        # iterate over all images
        for image_id in tqdm(self.image_id_list, desc='Testing'):
            #try:
                first = True
                prediction_labels_np = None
                prediction_max_value_np = None
                input_image = None
                # iterate over all valid landmarks
                for landmark_id in self.valid_landmarks[image_id]:
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    if first:
                        input_image = dataset_entry['datasources']['image']
                        prediction_labels_np = np.zeros(list(reversed(input_image.GetSize())), dtype=np.uint8)
                        prediction_max_value_np = np.ones(list(reversed(input_image.GetSize())), dtype=np.float32) * 0.5
                        first = False

                    image, prediction, transformation = self.test_full_image(dataset_entry)
                    del dataset_entry

                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    max_index = transformation.TransformPoint(np.array(self.image_size, np.float64) * np.array(self.image_spacing, np.float64))

                    if self.save_output_images:
                        utils.io.image.write_multichannel_np(image, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_input.mha'), output_normalization_mode='min_max', sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_prediction.mha'), output_normalization_mode=(0, 1), sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    del image
                    prediction = prediction.astype(np.float32)
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                                   output_spacing=self.image_spacing,
                                                                                                   channel_axis=channel_axis,
                                                                                                   input_image_sitk=input_image,
                                                                                                   transform=transformation,
                                                                                                   interpolator='cubic',
                                                                                                   output_pixel_type=sitk.sitkFloat32)
                    del prediction
                    #del transformation
                    prediction_resampled_np = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    if self.save_output_images:
                        utils.io.image.write_multichannel_np(prediction_resampled_np, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_prediction_resampled.mha'), output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=prediction_resampled_sitk[0].GetSpacing(), origin=prediction_resampled_sitk[0].GetOrigin())
                    bb_start = np.floor(np.flip(origin / np.array(input_image.GetSpacing())))
                    bb_start = np.maximum(bb_start, [0, 0, 0])
                    bb_end = np.ceil(np.flip(max_index / np.array(input_image.GetSpacing())))
                    bb_end = np.minimum(bb_end, prediction_resampled_np.shape - np.ones(3))  # bb is inclusive -> subtract [1, 1, 1] from max size
                    #print(bb_start, bb_end)
                    #bb_start, bb_end = utils.np_image.bounding_box(prediction_resampled_np)
                    slices = tuple([slice(int(bb_start[i]), int(bb_end[i]+1)) for i in range(3)])
                    prediction_resampled_cropped_np = prediction_resampled_np[slices]
                    if filter_largest_cc:
                        prediction_thresh_cropped_np = (prediction_resampled_cropped_np > 0.5).astype(np.uint8)
                        largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_cropped_np)
                        prediction_thresh_cropped_np[largest_connected_component == 1] = 0
                        prediction_resampled_cropped_np[prediction_thresh_cropped_np == 1] = 0
                    prediction_max_value_cropped_np = prediction_max_value_np[slices]
                    prediction_labels_cropped_np = prediction_labels_np[slices]
                    prediction_max_index_np = utils.np_image.argmax(np.stack([prediction_max_value_cropped_np, prediction_resampled_cropped_np], axis=-1), axis=-1)
                    prediction_max_index_new_np = prediction_max_index_np == 1
                    prediction_max_value_cropped_np[prediction_max_index_new_np] = prediction_resampled_cropped_np[prediction_max_index_new_np]
                    prediction_labels_cropped_np[prediction_max_index_new_np] = self.landmark_mapping[int(landmark_id)]
                    prediction_max_value_np[slices] = prediction_max_value_cropped_np
                    prediction_labels_np[slices] = prediction_labels_cropped_np
                    del prediction_resampled_sitk

                # delete to save memory
                del prediction_max_value_np
                center_points = self.getCenterpoints(prediction_labels_np)
                center_points = center_points[:, [2, 1, 0, 3]]
                num_stdev = self.getDistances(center_points * np.append(input_image.GetSpacing(), 1))
                # print(num_stdev)
                iqr_test = np.zeros(num_stdev.shape)
                q1 = np.nanquantile(num_stdev, 0.25)
                q3 = np.nanquantile(num_stdev, 0.75)
                num_stdev[np.isnan(num_stdev)] = np.nanquantile(num_stdev, 0.5)
                iqr = q3 - q1
                for k in np.where(num_stdev > q3 + (1.5 * iqr))[0]:
                        iqr_test[k] = num_stdev[k] - (q3 + (1.5 * iqr))
                clf = joblib.load('skipped_vert.joblib')
                prob = clf.predict_proba(iqr_test.reshape(-1, 1))*100
                loc = np.argmax(prob, axis=1)
                probability_skip = np.round(prob[:,1], 1)
                probability_skip[iqr_test==0] = 0
                threshold = 100
                # print(iqr_test)
                # print(probability_skip)
                if np.any(probability_skip == threshold) and check:
                    print('\nFound skipped vertebrae\n')
                    del prediction_labels_np
                    locations = np.where(probability_skip == threshold)[0]
                    print(f'locations: {locations}')
                    landmarks = pd.read_csv(os.path.join(self.setup_folder, 'vertebrae_localization/landmarks.csv'), header=None, keep_default_na= False).to_numpy()[0]
                    valid_landmarks = pd.read_csv(os.path.join(self.setup_folder, 'vertebrae_localization/valid_landmarks.csv'), header=None).to_numpy()[0]
                    shift = 0

                    for loc in locations:
                        loc += shift
                        idx1 = np.where(valid_landmarks == 7)[0]
                        if len(idx1) == 0:
                            idx1 = np.array([1])
                        idx2 = np.where(valid_landmarks == 19)[0]
                        if len(idx2) == 0:
                            idx2 = np.array([len(valid_landmarks)])
                        l_idx1 = (7 * 3) + 1
                        l_idx2 = (19 * 3) + 1
                        tmp_valid_landmarks = np.split(valid_landmarks, [idx1.item(), idx2.item()])
                        tmp_landmarks = np.split(landmarks, [l_idx1, l_idx2])
                        # loc += shift
                        i = loc*3 + 1
                        if np.any(landmarks[i+3:i+6]=='nan'):
                            point = (landmarks[i:i+3] + landmarks[i+6:i+9])/2
                        else:
                            point = (landmarks[i:i+3] + landmarks[i+3:i+6])/2
                        if loc < 6 or (loc <7 and tmp_landmarks[0][1]=='nan'):
                            if tmp_landmarks[0][-1]=='nan':
                                tmp_landmarks[0] = np.insert(tmp_landmarks[0], i+3, point)[:-3]
                                tmp_valid_landmarks[0] = np.append(tmp_valid_landmarks[0], tmp_valid_landmarks[0][-1] + 1)
                            else:
                                tmp_landmarks[0] = np.delete(np.insert(tmp_landmarks[0], i+3, point), range(1,4))
                                tmp_valid_landmarks[0] = np.hstack([tmp_valid_landmarks[0][0], tmp_valid_landmarks[0][1]-1, tmp_valid_landmarks[0][1:]])
                        elif loc < 19:
                            i2 = (loc - 7) * 3
                            if len(tmp_valid_landmarks[2]) != 0:
                                if tmp_valid_landmarks[1][-1] + 1 == tmp_valid_landmarks[2][0]:
                                    tmp_landmarks[1] = np.insert(tmp_landmarks[1], i2 + 3, point)
                                    tmp_landmarks[2][-3:] = tmp_landmarks[1][-3:]
                                    tmp_landmarks[1] = tmp_landmarks[1][:-3]
                                    tmp_valid_landmarks[2] = np.insert(tmp_valid_landmarks[2], len(tmp_valid_landmarks[2]), 25)
                                else:
                                    tmp_landmarks[1] = np.insert(tmp_landmarks[1], i2 + 3, point)[:-3]
                                    tmp_valid_landmarks[1] = np.append(tmp_valid_landmarks[1], tmp_valid_landmarks[1][-1] + 1)
                            else:
                                tmp_landmarks[1] = np.insert(tmp_landmarks[1], i2 + 3, point)[:-3]
                                tmp_valid_landmarks[1] = np.append(tmp_valid_landmarks[1], tmp_valid_landmarks[1][-1] + 1)
                        else:
                            i2 = (loc - 19) * 3
                            tmp_landmarks[2] = np.insert(tmp_landmarks[2], i2 + 3, point)[:-3]
                            tmp_valid_landmarks[2] = np.append(tmp_valid_landmarks[2], tmp_valid_landmarks[2][-1] + 1)
                        shift += 1
                        valid_landmarks = np.concatenate(tmp_valid_landmarks)
                        landmarks = np.concatenate(tmp_landmarks)
                    pd.DataFrame(landmarks).T.to_csv(os.path.join(self.setup_folder, 'vertebrae_localization/landmarks.csv'), index=False, header=False)
                    pd.DataFrame(valid_landmarks).T.to_csv(os.path.join(self.setup_folder, 'vertebrae_localization/valid_landmarks.csv'), index=False, header=False)
                    return True
                else:
                    prediction_labels = utils.sitk_np.np_to_sitk(prediction_labels_np)
                    prediction_labels.CopyInformation(input_image)
                    del prediction_labels_np
                    utils.io.image.write(prediction_labels, self.output_folder_handler.path(image_id + '_seg.nii.gz'))
                    if self.save_output_images:
                        prediction_labels_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(prediction_labels, [1.0, 1.0, 1.0], 'nearest'))
                        prediction_labels_resampled = np.flip(prediction_labels_resampled, axis=0)
                        utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg.png'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                        utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg_rgb.mha'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                        input_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(input_image, [1.0, 1.0, 1.0], 'linear'))
                        input_resampled = np.flip(input_resampled, axis=0)
                        utils.io.image.write_multichannel_np(input_resampled, self.output_folder_handler.path('output', image_id + '_input.png'), output_normalization_mode='min_max', image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)

                    del prediction_labels
                    return False
        # return False
            #except:
                #print('ERROR predicting', image_id)
                #pass


class dotdict(dict):
    """
    Dict subclass that allows dot.notation to access attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    # Set hyperparameters, which can be overwritten with a W&B Sweep
    hyperparameters = dotdict(
        load_model_filenames=parser_args.model_files,
        image_folder=parser_args.image_folder,
        setup_folder=parser_args.setup_folder,
        output_folder=parser_args.output_folder,
        num_filters_base=96,
        activation='lrelu',
        model='unet',
        num_levels=5,
        spacing=1.0,
        cv='inference'
    )
    with MainLoop(hyperparameters) as loop:
        loop.init_model()
        loop.init_output_folder_handler()
        loop.init_checkpoint()
        loop.init_datasets()
        print('Starting main test loop')
        restart = loop.test(True)
        if restart:
            loop.init_datasets()
            print('Restarting main test loop')
            restart = loop.test(False)


