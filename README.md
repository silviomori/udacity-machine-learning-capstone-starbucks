
### Udacity Machine Learning Engineer Nanodegree Program

# Capstone Project

## Starbucks Capstone Challenge

### Introduction

This is the final project of the [Udacity Machine Learning Engineer Nanodegree Program](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).


This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

### Included in this repository
- DataExploration.ipynb - notebook to present data visualization
- FeatureEngineering.ipynb - notebook to generate features
- ModelTraining.ipynb - notebook to train the models proposed
- SystemDemonstration.ipynb - notebook to demonstrate how the final model works
- data.zip - file containing the datasets
- linear_classifier.pt - file containing the weights of the benchmark model
- models.py - code for the neural networks
- proposal.pdf - document defining Capstone Project proposal
- report.pdf - a comprehensive report containing all the details concerning this Capstone Project
- README.md - this file
- recurrent_classifier.pt - file containing the weights of the final model


### Setting up the environment

##### Dataset
After downloading or cloning this repository, unzip the file `data.zip` in the root of the project. This will create a `data` folder containing three files:  

- portfolio.json
- profile.json
- transcript.json

##### Libraries
This project is developed in Python 3.6.  
You will need install some libraries in order to run the code.  
Libraries and respective version are:  

- jupyter 1.0.0
- numpy 1.17.4
- pandas 1.0.0
- matplotlib 3.1.1
- pytorch 1.3.1
- scikit-learn 0.22.1

