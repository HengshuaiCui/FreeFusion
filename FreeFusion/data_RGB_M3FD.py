import os
from dataset_RGB_M3FD import DataLoaderTrain, DataLoaderTest

def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)

def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir)
