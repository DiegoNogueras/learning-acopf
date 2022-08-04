# learning-acopf
A project for implementing the work done in the paper "Learning Optimal Solutions for Extremely Fast AC Optimal Power Flow"

## Introduction

In this repository, you will find various folders that contain Python scripts and Jupyter Notebooks that implement different steps to implement the work done in the previously mentioned paper. The following serves as an outline on how to properly and in which order execute the Python scripts and Jupyter Notebooks.

## Dependecies

To use the scripts described below, the following Python packages need to be installed for everything to work correctly. Download the most recent versions:

- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn
- Tensorflow
- Keras
- [Egret](https://github.com/grid-parity-exchange/Egret)
- PYPOWER

## Data Generation Process

To generate the training data for a particular IEEE bus network, the following steps were taken:

1. Copy the desired case file information from MATPOWER's documentation page. For example, in the case of the IEEE 118-bus system, the information in [this reference page](https://matpower.org/docs/ref/matpower5.0/case118.html) was copied.
2. Since the information copied from this documentation contains line numbers, they are also copied. To remove them, the `remove_numbers.py` script under the folder `reformatting_data` is executed. For more information on executing this script, see [Removing Numbers Script](#removing-numbers-script).
3. Finally, inside the `data_generation` folder, a script called `data_generation.py` is to be executed in order to generate all training samples, making sure that they are feasible ones. For more information on executing this script, see [Data Generation](#data-generation).

There exists different ways to acquire the data for the various IEEE bus systems, however, we found this method to be more straight-forward. The only nuisance is having to remove the line numbers after copying the information.

The following sections will serve to provide more detail on the steps outlined above.

### Removing Numbers Script

In this script, three basic actions are perfomed:

1. Read the file containing the copied IEEE bus system case data. *This file must be a **.txt** file.*
2. Then, the numbers are removed from each line of the text file by taking everything *but* the first 4 characters. As long as the user copies the line numbers from the first line all the way through the last line, and if the line numbers don't exceed 9999, this method will always work. This includes the first 1000 lines as numbering starts at 0001 in MATPOWER's description of the IEEE bus system case files.
3. Finally, the new lines are written to another file. With the line numbers removed, the user can now go to the next step of the Data Generation process.

There are only two variables in this script: `file_with_nums`, which holds the file path to the text file which contains the copied information, and `new_file`, which holds the name of the new file to be saved. 

#### File Naming Convention

It is important to note that since this implementation heavily levarages the [Egret package](https://github.com/grid-parity-exchange/Egret), it is recommended that the `new_file` variable be set to an appropiate name for compatability with Egret functions. Some examples are as follows:

- For IEEE 57-bus system: `case57.txt`
- For IEEE 118-bus system: `case118.txt`
- For IEEE 300-bus system: `case300.txt`

In general, the following format is to be followed. Let `N` be the number of busses in the IEEE bus system. Then, the name of the file should be `caseN.txt`

### Data Generation

Inside of this folder, there exists the `data_generation.py` script and a folder called `case118_data`, which only contains the `case118.txt` text file; the same file that was obtained in [Removing Numbers Script](#removing-numbers-script). The folder `case118_data` is kept as an example for the user and for separation of different case data files in case multiple IEEE bus systems are consider. These files are kept in different folders purely for organization purposes.

The `data_generation.py` script is simple to use. The only variable that needs to be changed here is `original_case`, which should contain the file path to the `caseN.txt` file inside the `caseN_data` folder; where `N` represents the number of busses for that IEEE bus system, as explained in [File Naming Convention](#file-naming-convention). The other variable that could be changed is `NSamples`, which contains the number of samples to be generated. However, in the paper from which this implementation is modeled after, this number is 100,000 samples. Therfore, it should be kept as is.

The script takes care of every aspect of the data generation process and, when the number of samples is divisible by 100, it saves the current samples to a file named `caseN_data.csv` to avoid losing any feasible samples if something interrupts the execution of the script, as it can take a lot of time to generate all of the samples.

## Learning AC Optimal Power Flow

To learn the mapping between the load demands across all busses and the optimal generator setpoints, the following steps were taken:

1. The script found inside the `data_preprocessing` folder called `train_model.py` is executed. For more information on executing this script, see [Train Model Script](#train-model-script)
2. The script `solve_power_flow.py` found in the `data_preprocessing` folder can be executed to obtain the predictions for the optimal generator setpoints and, from this, solve the power flow equations to obtain the remaining values. For more information on executing this script, see [Solver Power Flow Script](#solve-power-flow-script).
3. An additional folder named `analyzing_resuls` with a script for analyzing the results is also provided, called `analyzing_results.py`. For more information on the contents of this folder, see [Analyzing Results Script](#analyzing-results-script).

All of the scripts in this section levarge another python script called `helper_functions.py` which contains functions that assist in the data preprocessing, model training, model evaluation, among other things.

### Train Model Script

The first script that should be executed if doing this process for the first time, would be the `train_model.py` script. This script performs the necessary data preprocessing, trains the model, and prints an evaluation of the model. The only variable here that may need changing is the `generated_data` variable which contains the file path to the `.csv` file generated in the [data generation process](#data-generation-process).

### Solve Power Flow Script

After training the model, or if a model has already been trained using the previous script, the `solve_power_flow.py` script is to be executed to obtain the final results. In short, this script is in charge of predicting the generator setpoints and solving the power flow equations one sample at a time. This is because we need to measure how much time each sample takes to be solved end-to-end to compare it with other methods. The only variable here that may need changing is the `generated_data` variable which contains the file path to the `.csv` file generated in the [data generation process](#data-generation-process). The results of this process is saved to a file named `predicted_caseN_data.csv`, with `N` being the number of busses for that IEEE bus system.

### Analyzing Results Script

This script implements the required evaluation metrics for evaluating the end-to-end process outlined in the previous sections. The only variables that need to be changed are: `caseN_data_path`, `predicted_caseN_data_path`, and `original_caseN_data_path`. These variables contain the file paths to the file which was generated in the [data generation process](#data-generation-process), the predicted data obtained from the [Solver Power Flow Script](#solve-power-flow-script), and the `caseN.txt` file obtained in the [Removing Numbers Script](#removing-numbers-script), respectively. The script prints the results of the different evaluation metrics.
