# DNN framework for quark/gluon discrimination

## Initial setup
First clone the repository.
```
$ git clone https://github.com/kimmokal/qgDNN
$ cd qgDNN/
```
Run a bash script to setup the necessary directories.
```
$ source dir_setup.sh
```
Create a virtual environment in which to install the required Python packages.
```
$ virtualenv venv   #Make sure that the environment is created for Python 2.7, not Python 3
$ source venv/bin/activate
$ pip install -r requirements.txt
```
In the beginning of each new session, be sure to activate the virtual environment.

## Preprocessing
__NOTE: In all of the scripts below, you will need to change the _workPath_ line to the path to your own working directory.__

The first preprocessing script divides the jets into seven different eta,pT bins and saves them found in the _data/binned/_ directory. Running the script can take quite a long time (an hour or two).
```
$ python data/preprocess_bins.sh
```

After the jets have been put into bins, they need to be preprocessed further for the DNNs. There are three DNN models available for training (found in the _models_ directory), and each requires its own preprocessing script.

```
$ python models/fNN/preprocessTuples_fNN.py
$ python models/deepJet/preprocessTuples_deepJet.py
$ python models/jetImages/preprocessTuples_jetImages.py
```

The scripts will save the preprocessed jets to _data/trainSets/_ and _data/testSets/_. In order to preprocess the jets for each eta,pT bin, you will need to change the bin manually in the code of each scripts.

## DNN training

When the preprocessing is done, you can now train the DNN models. A specific model is trained for each eta,pT bin.
```
$ python models/fNN/preprocessTuples_fNN.py
$ python models/deepJet/preprocessTuples_deepJet.py
$ python models/jetImages/preprocessTuples_jetImages.py
```

The trained models are saved to the _models/trainedModels/_ directory.

## Plotting
There are two scripts for plotting in the _plotter/_ directory. The first one plots the ROC curves for the different models.
```
$ python plotter/AUC_plotter.py
```
The plots are saved to the _plotter/plots/_ directory. The second script can be used to compare the ROC AUC in different bins.
```
$ python plotter/compareAUC_plotter.py
```
This script is crude and requires one to manually input the ROC AUC values in the code.
