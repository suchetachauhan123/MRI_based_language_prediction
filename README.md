# MRI_based_language_prediction
#Description of the folders are given below: 
#full_train.py is the code written in tensorflow to train the MRI images (which are already in binary format in Data directory). 
#testfile.py: The input to the testfile.py will be .nii format MRI image with just lesion information (because network is trained only on lesion images). This code will help to convert .nii format MRI file into specified format and will convert into binary matrix and after preprocessing, it will print the language severity score of Stroke patients.
# To execute the test file, simply load MRI .nii file and execute the "testfile.py". 'network' folder already has trained parameters saved in it.

python testfile.py
