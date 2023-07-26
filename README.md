# Aerial Fire Detection with Semantic Segmentation Research
Due to the increased popularity of Unmanned Aerial Vehicles (UAVs) for monitoring and predicting potential fire regions, there is a need for models that are able to analyze aerial imagery and run efficiently on devices with limited computing power. 

This project aims to investigate efficient real-time fire detection methods using semantic segmentation and evaluate the effectiveness of different models in identifying and segmenting fires in aerial images. This project also introduces a couple modifications on the original [ERFNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) model to improve the overall efficiency and segmentation speed.

This project utilizes the Fire Luminosity Airborne-based Machine Learning Evaluation dataset (FLAME) dataset, a widely recognized collection of labeled aerial imagery specifically designed for fire-related analysis, which is available [here](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) (you may need to reload the link). 

This project uses the same preprocessing methods as [this notebook](https://github.com/maidacundo/real-time-fire-segmentation-deep-learning/blob/main/Fire%20Segmentation%20Pipeline.ipynb), with different model implementations to test the effectiveness of different models for fire segmentation.

## Setup

### Requirements

Required packages are listed in the `requirements.txt` file.

### Download dataset

* [FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
  * Specifically, the images used are from *(9) Images for fire segmentation (Train/Val/Test) Images.zip)* and *(10) Masks annotation for fire segmentation (Train/Val/Test) Masks.zip)* under *'Dataset Files'*
