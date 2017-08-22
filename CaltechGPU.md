# Instructinos for running on Caltech GPUs
 * Clone our `DeepJet` repository as usual, but do not install `miniconda2`
 ```bash
 git clone git@github.com:DeepDoubleB/DeepJet
 ```
 * Add this line to your `~/.bashrc` file:
 ```bash
 export PATH="/home/jduarte/miniconda2/bin:$PATH"
 ```
 * Log out and log back in (or type `bash`)
 * Setup the environment and make the modules
 ```bash 
 cd ~/DeepJet/environment/
 source gpu_env.sh
 cd ../modules
 make 
 ```
 * Check what GPUs are available by typing 
 ```bash
 nvidia-smi
 ```
 * You should get an output like the following, which means no GPUs are being used.
 ```
 Tue Aug 22 09:25:53 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 0000:04:00.0     Off |                  N/A |
| 27%   29C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 0000:05:00.0     Off |                  N/A |
| 27%   28C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 1080    Off  | 0000:06:00.0     Off |                  N/A |
| 27%   31C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 1080    Off  | 0000:07:00.0     Off |                  N/A |
| 27%   27C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   4  GeForce GTX 1080    Off  | 0000:0B:00.0     Off |                  N/A |
| 27%   26C    P8    10W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  GeForce GTX 1080    Off  | 0000:0C:00.0     Off |                  N/A |
| 27%   29C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   6  GeForce GTX 1080    Off  | 0000:0D:00.0     Off |                  N/A |
| 27%   28C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   7  GeForce GTX 1080    Off  | 0000:0E:00.0     Off |                  N/A |
| 27%   30C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
* As long as there's at least one free GPU you should be able to run a training on it.
* To run a (shortish) training in the background and save the output to a log file do the following. Edit the file `~/DeepJet/environment/train_deepdoubleb_Resnet.py` to only run for 5 epochs. Then do:
```bash
cd ~/DeepJet/environment/
python train_deepdoubleb_Resnet.py /bigdata/shared/BumbleB/convert_deepDoubleB_init_train_val/dataCollection.dc train_Resnet_sv/  2>&1 | tee train_Resnet_sv.log
```
* This should start the training. If you want to check that you're using the GPUs do `nvidia-smi`:
```
Tue Aug 22 09:32:55 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 0000:04:00.0     Off |                  N/A |
| 27%   31C    P2    37W / 180W |   7747MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 0000:05:00.0     Off |                  N/A |
| 27%   28C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 1080    Off  | 0000:06:00.0     Off |                  N/A |
| 27%   31C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 1080    Off  | 0000:07:00.0     Off |                  N/A |
| 27%   26C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   4  GeForce GTX 1080    Off  | 0000:0B:00.0     Off |                  N/A |
| 27%   27C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  GeForce GTX 1080    Off  | 0000:0C:00.0     Off |                  N/A |
| 27%   29C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   6  GeForce GTX 1080    Off  | 0000:0D:00.0     Off |                  N/A |
| 27%   28C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   7  GeForce GTX 1080    Off  | 0000:0E:00.0     Off |                  N/A |
| 27%   30C    P8     9W / 180W |      0MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     19011    C   python                                        7745MiB |
+-----------------------------------------------------------------------------+
```
