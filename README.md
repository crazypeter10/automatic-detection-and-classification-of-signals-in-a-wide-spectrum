# Signal Classification with CNN

This project is focused on the **automatic detection and classification of signals** using a Convolutional Neural Network (CNN). The model is trained to distinguish between different types of synthetic signals: sinusoidal, noisy, and interference. This classification task can have applications in fields such as telecommunications, biomedical signal processing, radar, and security surveillance.

## Project Structure

- `data/`: Contains the training and testing data folders.
  - `train/`: Training dataset with subfolders for each class (`sinusoidal`, `noisy`, `interference`).
  - `test/`: Testing dataset with subfolders for each class (`sinusoidal`, `noisy`, `interference`).
- `data_generation.py`: Script for generating synthetic signal data and saving spectrograms as images.
- `dataset.py`: Contains functions for loading data and creating data loaders.
- `model.py`: Defines the architecture of the CNN model used for signal classification.
- `train.py`: Script to train the CNN model on the generated data.
- `evaluate.py`: Script to evaluate the model and display results, including accuracy, confusion matrix, and classification examples.
- `model.pth`: Saved model file after training.

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- matplotlib
- torchvision

To install the required packages, run:
```bash
pip install torch torchvision scikit-learn matplotlib


Data Generation
The data_generation.py script generates synthetic data by creating spectrograms of different types of signals:

Sinusoidal: Simple periodic signals.
Noisy: Signals with added noise.
Interference: Signals with overlapping frequencies simulating interference.
Run the following command to generate training and testing data:

bash
Copy code
python data_generation.py
This will create folders under data/train/ and data/test/ with the generated spectrogram images for each class.

Training the Model
The train.py script trains the CNN model on the generated dataset.

To start training, run:

bash
Copy code
python train.py
This will train the model for a specified number of epochs and save the trained model as model.pth.

Evaluating the Model
The evaluate.py script evaluates the model on the test dataset. It provides three main outputs:

Accuracy: The overall accuracy of the model on the test set.
Confusion Matrix: A graphical representation of true vs. predicted classifications for each class.
Examples of Correct and Incorrect Classifications: A visual comparison of some correctly and incorrectly classified samples to help understand where the model performs well and where it struggles.
Run the following command to evaluate the model:

bash
Copy code
python evaluate.py
Results
The evaluation script will display:

Confusion Matrix: This shows the model’s classification performance for each class. It helps identify which classes the model may be confusing with others.
Examples of Correct and Incorrect Classifications: Provides insight into how the model interprets different spectrograms, allowing a better understanding of any misclassifications.
Sample Results
Accuracy: Typically, the model achieves around 64% accuracy, depending on the generated data and training parameters.
Confusion Matrix: Reveals that the model performs well on certain classes (e.g., sinusoidal) but may struggle with interference due to overlapping frequencies.
Classification Examples: Helps visualize how well the model distinguishes each signal type, showing both correct and incorrect predictions.
Possible Improvements
Increasing Data Variety: Adding more complex examples, especially for the interference class, to improve model generalization.
Model Architecture Adjustments: Adding more convolutional layers or increasing the number of filters to capture more details in the spectrograms.
Parameter Tuning: Adjusting learning rate, batch size, and number of epochs to achieve better convergence.
License
This project is open-source and available for modification and enhancement.

Acknowledgements
This project was created as part of a study on Computational Intelligence in Software Engineering. The methodology and techniques applied here are foundational in the field of signal classification and can be extended to real-world data for practical applications.
