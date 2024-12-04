# **Spectra Data Toolkit (SWE4S Group 10)**

This is an assortment of software aimed towards making it easier to perform data analysis of absorption spectroscopy data, particularly when it comes to dual-comb spectroscopy. This repository seeks to achieve the following:
- Create a user-friendly GUI with which to interface with existing pldspectrapy fitting code
- Enable the creation of pre-computed lookup tables of spectral absorption data for inputted molecules at various inputted parameters
- Utilize these lookup tables to accelerate the fitting process of spectral absorption data


## **Table of Contents**
1. [Overview](#overview)
2. [Features and Usage](#features)
3. [Installation](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Steps](#installation)
4. [Acknowledgments](#acknowledgments)

## **Features and Usage**

### **GUI**

### **Lookup Table Creation**
- Feature 2
- Feature 3

Include a screenshot or visual representation if possible.


## **Installation**

### **Prerequisites**
- Python
- [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
- Git
- (Optional) PyCharm (or another IDE of your choice)

### Steps

1. Clone the repository and enter the directory:

```bash
git clone https://github.com/ErickGatica/Software_Eng_Project.git
cd Software_Eng_Project
```

2. Create a conda environment from the `environment.yaml` file:

```bash
conda env create -f environment.yaml
```
_Note if you want to name this environment somethign different, you can use the 
`--name` flag to overwrite the name in the environment file_

If you've already created the environment and just want to update it, you can use the following command:

```bash
conda env update -f environment.yaml
```

This will create a new conda environment named `spectra_software_env` and install the 
specified dependencies into it.

3. Activate the conda environment:

```bash
conda activate spectra_software_env
```

## **Acknowledgments**

The developers of this code with like to thank Eli Miller and Sean Coburn for helpful discussions.