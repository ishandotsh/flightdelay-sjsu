# Flight Delay Prediction Project

## Team Members
- Ishan Sharma
- Shashank Cuppala
- Yug Harshadkumar Patel

## Project Overview
This project aims to predict flight delays using machine learning techniques. We developed a web application that allows users to input flight details and receive a probability of delay.

## Dataset
- Source: Airlines Dataset
- Instances: 539,383
- Features:
  1. Airline
  2. Flight
  3. AirportFrom
  4. AirportTo
  5. DayOfWeek
  6. Time
  7. Length
  8. Delay

## Methodology
We explored multiple machine learning models:

### 1. Multi-Layer Perceptron (MLP)
- Accuracy: 65.66%
- Preprocessing:
  - StandardScaler for numerical features
  - BinaryEncoder for categorical features
- Architecture:
  - Hidden Layers: (512, 512, 256)
  - Activation: ReLU
  - Solver: Adam

### 2. Logistic Regression
- Saved model with accuracy of 65.49%
- Used for the final web application prediction

## Web Application
- Built with Flask
- Allows users to input flight details
- Provides delay probability prediction
- Simple and intuitive interface

## Future Improvements
- Incorporate more features
- Collect more recent flight data
- Experiment with advanced ensemble methods
- Improve model interpretability

## Challenges
- Handling categorical variables
- Balancing model complexity and performance
- Creating a user-friendly interface

## Conclusion
Our flight delay prediction model provides a practical tool for travelers to anticipate potential flight delays.
