# data

This folder contains the data for testing the model.

As example we consider a subset of the PTB-XL dataset.
The file `ptbxl_selected.csv` contains the IDs and labels of the selected ECGs.
Once the PTB-XL data is downloaded and the test data is built with `python build_test_dataset.py`, the test data set is stored in `test_data.hdf5`.
You can adapt the code in `build_test_dataset.py` to build your own test data set.
