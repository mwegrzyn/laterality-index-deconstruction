#!/bin/bash

# this script allows to transform notebooks into .py files
# which can then be run as a python module
# To achieve this, makeIpynbScripts uses jupyter nbconvert
# to turn ipynb's into py's, but omits all cells which have
# been tagged with "hide-cell", so all the interim outputs of
# the notebooks are omitted. 

echo "copy all models into scr"
cp -f ../models/* ../src/models

echo "copy external data into scr"
cp -f ../data/external/* ../src/data/external

echo "convert scripts into py and put into src"
./makeIpynbScripts.sh ../notebooks/04-mw-get-lat-data.ipynb ../src/data/make_dataset.py 
./makeIpynbScripts.sh ../notebooks/05-mw-compute-2d-and-li.ipynb ../src/features/build_features.py 
./makeIpynbScripts.sh ../notebooks/09-mw-apply-classifier-to-data.ipynb ../src/models/predict_model.py
./makeIpynbScripts.sh ../notebooks/12-mw-single-patient-results.ipynb ../src/visualization/visualize.py 

