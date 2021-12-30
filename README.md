# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio.
Download the starter files.
Download/Make the dataset available.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
I setup the debugger and profiler using the hooks within the training script. Once injected, during the training phase all respective information was collected and then analyzed via the profiler report.

### Results
GPU/CPU utilization wasn't being fully utilized. This could be due to the fact we were using a pretrained model and only training the final two fully connected layers. Additionally we used the GPU accelerated instance which might have been overkill.


## Model Deployment
Call `predictor.predict(json.dumps(request_dict), initial_args={"ContentType": "application/json"})` with the request dict being an object with a `url` key and value being any image in s3.

Once you have the response you can get the dog breed class by calling `np.argmax(response, 1)` which you can then take the 1st element and compare against the 133 classes.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
