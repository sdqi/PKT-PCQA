# Data Preprocessing

## 1. Key Points Extraction
First, place the point clouds that require quality prediction in the same directory. Modify the `root` variable in the `key_points_extract.m` script under the `data_preparation` folder to this path, and also change the `out_root` to the corresponding output path. Run the script to obtain the point clouds with extracted key points.
Additionally, obtain the corresponding Excel spreadsheet for subsequent data preparation.

## 2. Supplement MOS Labels in Excel Spreadsheet
Fill in the MOS labels for the corresponding point clouds in the Excel spreadsheet.

## 3. MOS Normalization and Label Classification
Run the `mos_normal_label_txt.py` script under the `data_preparation` folder to perform MOS normalization and generate 'good', 'fair', and 'bad' classification labels. This script also splits the data into training and testing sets for classification and quality prediction.

## 4. KNN Processing
Run the `knn.py` script under the `data_preparation` folder to perform KNN processing on the point clouds with extracted key points.

# Training

## 1. Classification Training
Modify and run the `train_cls.py` script to perform classification training.

## 2. Quality Prediction Training
Modify and run the `train_pred.py` script to perform quality prediction training.

# Testing

## 1. Classification Testing
Modify and run the `test_cls.py` script to perform classification testing.

## 2. Quality Prediction Testing
Modify and run the `test_pred.py` script to perform quality prediction testing.
