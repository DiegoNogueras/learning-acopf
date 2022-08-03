# learning-acopf
A project for implementing the work done in the paper "Learning Optimal Solutions for Extremely Fast AC Optimal Power Flow"

## Introduction

In this repository, you will find various folders that contain Python scripts and Jupyter Notebooks that implement different steps to implement the work done in the previously mentioned paper. The following serves as an outline on how to properly and in which order execute the Python scripts and Jupyter Notebooks.

## Data Generation

To generate the training data for a particular IEEE bus network, the following steps were taken:

1. Copy the desired case file information from MATPOWER's documentation page. For example, in the case of the IEEE 118-bus system, the information in [this reference page](https://matpower.org/docs/ref/matpower5.0/case118.html) was copied.
2. Since the information copied from this documentation contains line numbers, they are also copied. To remove them, the **remove_numbers.py** script under the folder **reformatting_data** is executed. For more information on executing this script, see [Removing Numbers Script](#removing-numbers-script).
3. After this, inside the **data_generation** folder, a script called **data_generation.py** is to be executed in order to generate all training samples, making sure that they are feasible ones. For more information on executing this script, see [Data Generation Script](#data-generation-script).
4. Finally, the script **solver_power_flow.py** found in the **data_preprocessing** folder can be executed to obtain the predictions. For more information on executing this script, see [Solver Power Flow Script](#solve-power-flow-script).
5. An additional folder with scripts for analyzing the results is also provided, called **analyzing_results**. For more information on the contents of this folder, see [Analyzing Results](#analyzing-results).

There exists different ways to acquire the data for the various IEEE bus systems, however, we found this method to be more straight-forward. The only nuisance is having to remove the line numbers after copying the information.

The following sections will serve to provide more detail on the steps outlined above.

### Removing Numbers Script

In this script, three basic actions are perfomed:

1. Read the file containing the copied IEEE bus system case data. *This file must be a **.txt** file.*
2. Then, the numbers are removed from each line of the text file by taking everything *but* the first 4 characters. As long as the user copies the line numbers from the first line all the way through the last line, and if the line numbers don't exceed 9999, this method will always work. This includes the first 1000 lines as numbering starts at 0001 in MATPOWER's description of the IEEE bus system case files.
3. Finally, the new lines are written to another file. With the line numbers removed, the user can now go to the next step of the Data Generation process.

There are only two variables in this script: `file_with_nums`, which holds the file path to the text file which contains the copied information, and `new_file`, which holds the name of the new file to be saved. It is important to note that since this implementation heavily levarages the [Egret package](https://github.com/grid-parity-exchange/Egret), it is recommended that the `new_file` variable be set to an appropiate name for compatability with Egret functions. Some examples are as follows:

- For IEEE 57-bus system: `case57.txt`
- For IEEE 118-bus system: `case118.txt`
- For IEEE 300-bus system: `case300.txt`

In general, the following format is to be followed. Let `N` be the number of busses in the IEEE bus system. Then, the name of the file should be `caseN.txt`

### Data Generation Script



### Solve Power Flow Script

### Analyzing Results
