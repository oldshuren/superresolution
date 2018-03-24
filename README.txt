First you need to install tensorflow, instructions is here https://www.tensorflow.org/install/ Please note you need install 64 bit python on Windows!

** Generating Training and Evaluating Data

There are two ways to accomplish this,

* Use matlab, here are the steps

1) First run data/matlab-gen-data/GenerateTrainAndTestData_V1.m under matlab, it will generate two files TrainData.txt and TestData.txt.

2) The run data/preprocess-matlab-data.py to generate TF record for tensorflow as

$ python preprocess-matlab-data.py --train_data path_to_TrainData.txt --test_data path_to_TestData.txt

There genenerated files are under folder generated which can be change by --output_dir option

* Use Octave,

1) Install Octave and put bin directory of Octave in the PATH.
2) Install two python packages oct2py and scipy, it can be done with pip

pip install --upgrade oct2py scipy

3) run data/gen_data.py

python data/gen_data.py

** Train and Predict

At the top level folder, run

python train_single.py

After it finishes, it will print out folder where the model is saved. The you can

python predic_single.py --model_dir folder_model_is_saved --input_data path_to_TestData.txt

You can change the structure of the neural net, such as the number of hidden layers, the size of the hidden layer and where the model is saved by providing command options to train_single.py. You can get all the options' help by,

python train_single.py --help
