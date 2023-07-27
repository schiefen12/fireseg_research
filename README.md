# Aerial Fire Detection with Semantic Segmentation Research
Due to the increased popularity of *Unmanned Aerial Vehicles (UAVs)* for monitoring and predicting potential fire regions, there is a need for models that are able to analyze aerial imagery and run efficiently on devices with limited computing power. 

This project aims to investigate efficient real-time fire detection methods using semantic segmentation and evaluate the effectiveness of different models in identifying and segmenting fires in aerial images. This project also introduces a couple modifications on the original [*ERFNet*](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) model to improve the overall efficiency and segmentation speed.

This project utilizes the Fire Luminosity Airborne-based Machine Learning Evaluation dataset (FLAME) dataset, a widely recognized collection of labeled aerial imagery specifically designed for fire-related analysis, which is available [here](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) (you may need to reload the link). 

This project uses the same preprocessing methods as [this notebook](https://github.com/maidacundo/real-time-fire-segmentation-deep-learning/blob/main/Fire%20Segmentation%20Pipeline.ipynb), with different model implementations to test the effectiveness of different models for fire segmentation.

## Setup

### Requirements
Required packages are listed in the `requirements.txt` file.
To install the required packages directly from the file into the notebook, you can run this command in a notebook cell:
```sh
pip install -r requirements.txt
```
You can ensure the required packages have been installed by running this command in a Jupyter Notebook cell:
```sh
pip freeze
```

These models were trained and tested on an IPython 7.31.1 kernel with a NVIDIA P100 GPU using Python 3.9.13 and CUDA version 11.2.1.

### Download dataset
* [FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
  * Specifically, the images used are from *(9) Images for fire segmentation (Train/Val/Test) Images.zip)* and *(10) Masks annotation for fire segmentation (Train/Val/Test) Masks.zip)* under *'Dataset Files'*

## Modified ERFNet Architecture
This project improved upon the segmentation speed and efficiency of the original *ERFNet* model by halving the number of `non-bt-1D` layers throughout the model and by changing the `1x3` convolutions within the `non-bt-1D` to `1x1` convolutions.

![](Images/Modified-ERFNet-Diagram.png)

## Usage
This project is set up as a Jupyter Notebook and all the necessary modules and functions are defined within the notebook.

## Dependencies
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [PyTorch](https://pytorch.org/get-started/pytorch-2.0/)
* [MatPlotLib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [Torchinfo](https://pypi.org/project/torchinfo/)
* [Torchvision](https://pypi.org/project/torchvision/)

## References
Li, Mengna, et al. "A Real-time Fire Segmentation Method Based on A Deep Learning Approach." *IFAC-PapersOnLine* 55.6 (2022): 145-150. url: https://www.sciencedirect.com/science/article/pii/S2405896322005055

Romera, Eduardo, et al. "Erfnet: Efficient residual factorized convnet for real-time semantic segmentation." *IEEE Transactions on Intelligent Transportation Systems* 19.1 (2017): 263-272. url: http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III* 18. Springer International Publishing, 2015. url: https://arxiv.org/abs/1505.04597

Yesilkaynak, Vahit Bugra, Yusuf H. Sahin, and Gozde Unal. "Efficientseg: An efficient semantic segmentation network." *arXiv preprint arXiv:2009.06469* (2020). url: https://arxiv.org/abs/2009.06469

