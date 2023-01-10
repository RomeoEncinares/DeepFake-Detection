# DeepFake-Detection

A project that involves using deep learning technologies such as Recurrent Neural Networks in detecting deepfakes.

## To train the model locally

1. Clone the project

```bash
  git clone https://github.com/RomeoEncinares/DeepFake-Detection
```

2. Go to the project directory

```bash
  cd DeepFake-Detection
```

3. Create a folder for the dataset

4. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/overview)

5. Extract the dataset to two folders

```
  test_videos
  train_sample_videos
```

4. Open Anaconda Navigator / Anaconda Prompt to create a conda environment. Use the appropriate Python version, Tensorflow version, cuDNN, and CUDA according to your hardware. Refer to the [Tensorflow](https://www.tensorflow.org/install/source_windows) documentations.

5. Open the environmen through terminal.

6. Install the dependencies

```bash
  pip install -r requirements.txt
```
6. Open Jupyter Notebook 

```bash
  jupyter notebook
```
7. Run the notebook

## Roadmap

- Improve the architecture

- Use a larger dataset

- Extract more features

