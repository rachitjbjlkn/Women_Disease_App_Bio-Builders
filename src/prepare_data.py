"""Utility script to generate or load all sample datasets."""
import sys, os
# make sure parent directory (project root) is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import data_processing


def main():
    ov = data_processing.get_ovarian_cyst_data()
    pc = data_processing.get_pcos_data()
    bc = data_processing.get_breast_cancer_data()
    print("Ovarian cyst data", ov.shape)
    print("PCOS data", pc.shape)
    print("Breast cancer data", bc.shape)

if __name__ == "__main__":
    main()
