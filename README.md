# Session 1

## Requirements

  - [Python3] 
  - [SimPy] discrete-event simulation library
  - [Matplotlib] or any other 2D plotting library

## Installation

Install python 3.6:

```sh
$ sudo apt-get update
$ sudo apt-get install python3.6
```

Check if installation succeeded, the following command should display your python version:
```sh
$ python3.6 --version
```
Install python packet management system PIP:

```sh
$ sudo apt-get install python3-pip
```

Install the libraries:

```sh
$ sudo pip3 install -U simpy matplotlib
```

## Homework 
- Write a SimPy program that simulates a Poisson packet arrival process with intensity \lambda = 15 packets per second
- What is the observed average packet rate ? 
- What is the 95% confidence interval ?

   [Python3]: <https://www.python.org/>
   [SimPy]: <https://simpy.readthedocs.io/en/latest/contents.html>
   [Matplotlib]: <https://matplotlib.org/>
   
   