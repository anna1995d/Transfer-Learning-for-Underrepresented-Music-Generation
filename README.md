# MusicVAE-TF2
Migrating MusicVAE from TF 1.x to TF 2.x

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
