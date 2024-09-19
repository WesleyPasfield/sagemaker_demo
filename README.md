# SageMaker Demo

### Purpose

The repo aims to walk through a simple demo to train a model using SageMaker, and then a seperate SageMaker job to generate predictions on a trained model leveraging the SageMaker SDK, specifically the SKLearn image.

A SageMaker notebook is used to orchestrate the model training & prediction generation, but the actual compute is happening in the emphemeral SageMaker jobs. The notebook also loads the final predictions generated.

The motivation is to separate compute out for different processes to optimize compute size for the required task, and remove issues with overloading a single instance from multiple people or processes working at the same time.

This operates in a world where ETL is largely happening upstream, and final model training files exists and are ready for modeling. But more model pre-processing could be integrated into this process as well.

### File Structure

The files included are as follows:

- `notebook_demo_example.ipynb` The Jupyter Notebook that orchestrates the model training & prediction computation, as well as reading the predictions from S3 into the notebook environment
- `train.py` The model training code that runs when the model training job executes
- `inference.py` The inference code that runs when the predictions are computed
- `get_data.py` A helper script that creates the datasets that we'll use in this demo. This is not integrated into the SageMaker process, just a means to make the training & test files used.

