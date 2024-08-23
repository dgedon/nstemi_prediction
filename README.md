# Classification of STEMI and NSTEMI from ECGs with deep models

Official repository for the paper, see [this link](https://www.nature.com/articles/s41598-022-24254-x):

`
Gustafsson, S., Gedon, D., Lampa, E. et al., Development and validation of deep learning ECG-based prediction of myocardial infarction in emergency department patients. Scientific Report*, 12, 19615 (2022).
`

This repository contains only the code for **testing** the model. It does not contain the code for training the model. To run the model, you need to have the trained model weights. The model weights are available upon request. Please contact Daniel Gedon, email: [daniel.gedon@it.uu.se](mailto:daniel.gedon@it.uu.se).

## Installation

This code is tested with Python 3.10.14. Install the required packages with

```bash
pip install h5py numpy pandas scikit-learn torch tqdm wfdb
```

## Download example data: PTB-XL

As example, we consider the PTB-XL dataset, see [here](https://physionet.org/content/ptb-xl/1.0.3/). We extract some ECGs from the PTB-XL dataset to build a test data set. Specifically, we manually inspected ECGs with and without STEMI and selected a total of 275 ECGs for testing. The IDs and labels are given in the file `data/ptbxl_selected.csv`.

To extract the test data set for usage in the code, follow these steps:

1. Download and extract the files

    ```bash
    wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip -d data
    mv data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 data/ptb-xl
    ```

2. Build the test data set

    ```bash
    python build_test_dataset.py
    ```

    Resulting in a file `data/test_data.hdf5` with the test data set.

You can adapt the code in `build_test_dataset.py` to build your own test data set.

## Run the model on data

To run the model on the test data set, use the following command:

```python
python main_test_nstemi.py
```

You can modify to run your own dataset with

```python
python main_test_nstemi.py \
    --input_data=PATHTOYOURDATA \
    --log_dir=PATHTOYOURLOGDIR 
```

The model will output the predictions and the corresponding labels in the log directory. The predictions are stored the file `logits.csv`.

## Evaluation of results

The test code above directly run the evaluation as well. If you want to evaluate the logits file, you can use the following command:

```python
python eval_results.py --log_file=logs/logits.csv
```

For the example PTB-XL data set, the output should be

```bash
Results: control vs MI (STEMI+NSTEMI)
ROC AUC: 0.9562
PR AUC: 0.9327
```

## Citation

Consider citing our paper if you find this work useful:

```bib
@article{Gustafsson2022,
  title = {Development and validation of deep learning ECG-based prediction of myocardial infarction in emergency department patients},
  volume = {12},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Gustafsson,  Stefan and Gedon,  Daniel and Lampa,  Erik and Ribeiro,  Ant\^onio H. and Holzmann,  Martin J. and Sch\"{o}n,  Thomas B. and Sundstr\"{o}m,  Johan},
  year = {2022},
  }
```
