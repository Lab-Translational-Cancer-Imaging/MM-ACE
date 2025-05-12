# Automated vertebrae identification and segmentation with anatomical coherence inspection
This repository contains the guidelines and code on how to use an automated vertebrae segmentation with additional anatomical coherence inspection based on the method from Madzia-Madzou et al. ([Automated vertebrae identification and segmentation with structural uncertainty analysis in longitudinal CT scans of patients with multiple myeloma](https://doi.org/10.5220/0008975201240133)). The goal of this repository is to make running the code as simple as possible. For this reason, a singularity container is utilized. The singularity container contains all the necessary files and has all the used Python packages installed. So, make sure you have a machine (preferably with a GPU) with singularity installed and let's segment some vertebrae.
 
## Usage
All you have to do is obtain the singularity container and run the code with the steps below. I will provide the bare bones here since how you can access your GPU varies between different machines.
1. Run the line `singularity pull mm_ace.sif library://djennifer/vertseg/mm_ace`
2. Activate the singularity container with access to your GPUs `singularity shell --nv mm_ace.sif`
3. Enter the following lines of code in your singularity container:
   
   `cd /MedicalDataAugmentationTool/bin/`

   `export PYTHONPATH=/MedicalDataAugmentationTool`

   `python start_cv.py --data_dir '[path data_directory]' --pipeline 'all' "$@"`

Make sure that your data directory contains other directories with the .nii.gz files inside of them. This is an example of the wanted structure.
```bash
data_directory
   ├───dir_scan_1
   │        ├───scan_1.nii.gz
   ├───dir_scan_2
   │        ├───scan_2.nii.gz
   ├───dir_scan_3
   │        ├───scan_3.nii.gz
   ...
```

## Guided manual evaluation

In the directory with all the segmentations, there is a tmp directory. Here you will find two .csv files. The first one, corrected.csv, keeps track of segmentations that are automatically corrected. The second file, uncertainty.csv, contains the uncertain locations or scans that require a check. The file includes a couple of columns that you might need to edit after reviewing the images:

```
file = segmentation file
locations = check if the vertebra after this vertebra id is segmented
score = how uncertain the model is about this location (we checked the scans if > 14)
review = put 1 if there is indeed a missed vertebra and 0 if not
manual_correction = is 1 if a missing section is detected and needs to be manually corrected
manually_corrected = put 1 after you have done the manual corrections
```

After the scans are reviewed and corrected you can run the review in singularity with: 

   `python start_cv.py --data_dir '[path data_directory]' --pipeline 'all' --review 'True' "$@"`

## Manual corrections
If you want to edit/add centerpoints to the segmentation, you need to navigate to the tmp/[file]/vertebrae_localization directory. Here you will find a file named: slicer_landmarks.mrk.json. You can load this into Slicer with your image and add a markup if needed. Make sure that the names of the points are corresponding to the vertebra id. Export the markups as a Table.csv and put it in the tmp/[file]/vertebrae_localization directory.

## Customizability
The folder `source/MedicalDataAugmentationTool` contains the original code from Payer et al. ([Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://doi.org/10.5220/0008975201240133)) with the additions. If you want to change the singularity container. Edit your wanted file and copy it into the correct location inside the singularity shell. If you want to use docker, follow Payer's instructions and copy the bin folder to add the anatomical coherence inspection.
