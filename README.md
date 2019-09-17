# runForest
A command line interface program to apply the RandomForest machine learning model for feature selection and classification of unknown data points

### Train model:

`python ./src/runForest.py -mode train -features example_files/training_data.txt -labels example_files/training_labels.txt -outpath example_files/runForest_out -seed 1234`

### Predict category labels for unknown data:

`python ./src/runForest.py -mode predict -features example_files/unknown_data.txt -trained_model example_files/runForest_out/trained_model_RF_10_4000_10_1234.pkl -outpath example_files/runForest_out -seed 1234`