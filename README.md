# Transfer Learning for Underrepresented Music Generation
This repository is a companion resource to my M.Sc. thesis of the same name as well as the paper at https://arxiv.org/abs/2306.00281. 
Doosti, Anahita, and Matthew Guzdial. "Transfer Learning for Underrepresented Music Generation." arXiv preprint arXiv:2306.00281 (2023).

## MusicVAE
This work builds on the code available for MusicVAE by Google Magenta. (See [repository](https://github.com/magenta/magenta))

Use the notebooks in the `\notebooks` to run experiments. (I recommend running on Colab.)

The Iranian folk music dataset is in `data\tfrecord\Persian\persian_100_v1`. They are seperated into 5 folds.

- fold_X_train.tfrecord: Contains every fold other than fold X. 
- fold_X_test.tfrecord: Contains fold X.

(X a number from 1 to 5.)

## Compute Canada Setup
1. Clone the repository [on a Cedar server]
2. Set the python module

   ```Shell
   [name@server ~]$ module load python/3
   ```
3. Create a virtual environment
  
   ```Shell
   [name@server ~]$ virtualenv --no-download MVAE2
   ```
4. Activate your newly created Python virtual environment.

   ```Shell
   [name@server ~]$ source tensorflow/bin/activate
   ```
5. Install TensorFlow in your newly created virtual environment using the following command.

   ```Shell
   (MVAE2) [name@server ~]$ pip install --no-index tensorflow==2.11
   ```
## Contact
This Github is not going to be actively maintained. Feel free to contact me for assistance and more info: anahita.doosti@ualberta.ca