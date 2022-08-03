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


### Removing Numbers Script

### Data Generation Script

### Solve Power Flow Script

### Analyzing Results
