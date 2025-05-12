import subprocess
import os
import sys
import argparse
from glob import glob
import pandas as pd

def parse_args():
    parser=argparse.ArgumentParser(description="segment vertebraes using MM-ACE")
    parser.add_argument("-p", "--pipeline", type=str, help='which pipeline')
    parser.add_argument("-d", "--data_dir", type=str, help='directory that contains the data')
    parser.add_argument("-c", "--check", type=str, default='True', help='perform the ACE check when true')
    parser.add_argument("-r", "--review", type=str, default='False', help='correct the manually reviewed scans')
    args=parser.parse_args()
    return args

def main():
    inputs=parse_args()
    base_image_folder = inputs.data_dir
    base_output_folder = os.path.join(base_image_folder, 'results_MM_ACE')
    base_intermediate_folder = os.path.join(base_image_folder, 'tmp')
    models_folder = '/models'

    if inputs.review == 'False':
        #pipeline = sys.argv[1:] if len(inputs.pipeline) > 1 else ['all']
        pipeline = inputs.pipeline
        print('Using pipeline: ', pipeline)

        all_image_folders = [os.path.split(path)[-1] for path in glob(os.path.join(base_image_folder, '*')) if os.path.isdir(path) and path != base_output_folder]
    else:
        pipeline = 'vertebrae_localization vertebrae_segmentation postprocessing'
        print('Using pipeline: ', pipeline)
        df_review = pd.read_csv(os.path.join(base_intermediate_folder, 'uncertainty.csv'), index_col=0)
        df_review = df_review[(df_review['review']==1) | (df_review['manually_corrected']==1)]
        if 'corrected' in df_review.columns:
            df_review = df_review[df_review['corrected']==0]
        all_image_folders = [row['file'].split('/')[-3] for index, row in df_review.iterrows()]
        all_image_folders = list(set(all_image_folders))

    for current_image_folder in sorted(all_image_folders):
        print('Processing folder ', current_image_folder)
    
        image_folder = os.path.join(base_image_folder, current_image_folder)
        output_folder = os.path.join(base_output_folder, current_image_folder)
        intermediate_folder = os.path.join(base_intermediate_folder, current_image_folder)
        
        preprocessed_image_folder = os.path.join(intermediate_folder, 'data_preprocessed')
        spine_localization_folder = os.path.join(intermediate_folder, 'spine_localization')
        spine_localization_model = os.path.join(models_folder, 'spine_localization')
        vertebrae_localization_folder = os.path.join(intermediate_folder, 'vertebrae_localization')
        vertebrae_localization_model = os.path.join(models_folder, 'vertebrae_localization')
        vertebrae_segmentation_folder = os.path.join(intermediate_folder, 'vertebrae_segmentation')
        vertebrae_segmentation_model = os.path.join(models_folder, 'vertebrae_segmentation')

        if 'preprocessing' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'preprocess.py',
                            '--image_folder', image_folder,
                            '--output_folder', preprocessed_image_folder,
                            '--sigma', '0.75'])
        if 'spine_localization' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_spine_localization.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', spine_localization_model,
                            '--output_folder', spine_localization_folder])
        if 'vertebrae_localization' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_vertebrae_localization.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', vertebrae_localization_model,
                            '--output_folder', vertebrae_localization_folder,
                            '--base_folder', base_intermediate_folder,
                            '--review', inputs.review])
        if 'vertebrae_segmentation' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_vertebrae_segmentation.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', vertebrae_segmentation_model,
                            '--output_folder', vertebrae_segmentation_folder,
                            '--base_folder', base_intermediate_folder,
                            '--check', inputs.check,
                            '--review', inputs.review])
        if 'postprocessing' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'cp_landmark_files.py',
                            '--landmark_folder', vertebrae_localization_folder,
                            '--output_folder', output_folder])
            subprocess.run(['python', 'reorient_prediction_to_reference.py',
                            '--image_folder', vertebrae_segmentation_folder,
                            '--reference_folder', image_folder,
                            '--output_folder', output_folder])


if __name__ == '__main__':
    main()