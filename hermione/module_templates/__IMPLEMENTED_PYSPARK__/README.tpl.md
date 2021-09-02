# Hermione Sagemaker

This notebook explains how to execute the Titanic project example


## Sagemaker

Our code is divided in three steps: Processor, Train and Inference. In the Processor step, we preprocessed the training, validation and inference data. The Train step receives the preprocessed training and validation data, and uses them to train and validate a new model. The Inference step receives the inference data and model, and generates the prediction for the data.

### Permitions

If you are running this code on a SageMaker notebook instance, do the following to provide IAM permissions to the notebook:

1. Open the Amazon [<b>SageMaker console</b>](https://console.aws.amazon.com/sagemaker/).
2. Select <b>Notebook instances</b> and choose the name of your notebook instance.
3. Under <b>Permissions and encryption</b> select the role ARN to view the role on the IAM console.
4. Under the <b>Permissions</b> tab, choose <b>Attach policies</b> and search for <b>AmazonS3FullAccess</b>.
5. Select the check box next to <b>AmazonS3FullAccess</b>.
6. Search for <b>AmazonSageMakerFullAccess</b> and <b>AWSStepFunctionsFullAccess</b> and select their check boxes.
7. Choose <b>Attach policy</b>. You will then be redirected to the details page for the role.
8. Copy and save the IAM role ARN for later use.

Next, we will create a new policy to attach.

12. Click <b>Attach policies</b> again and then <b>Create policy</b>.
13. Enter the following in the <b>JSON</b> tab:  

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "logs:CreateLogStream",
                "codebuild:DeleteProject",
                "codebuild:StartBuild",
                "s3:DeleteObject",
                "codebuild:CreateProject",
                "codebuild:BatchGetBuilds"
            ],
            "Resource": [
                "arn:aws:s3:::sagemaker-*/*",
                "arn:aws:codebuild:*:*:project/sagemaker-studio*",
                "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*"
            ]
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": [
                "logs:GetLogEvents",
                "s3:CreateBucket",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*:log-stream:*",
                "arn:aws:s3:::sagemaker*"
            ]
        },
        {
            "Sid": "VisualEditor2",
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "ecr:CreateRepository",
                "iam:ListRoles",
                "ecr:GetAuthorizationToken",
                "ecr:UploadLayerPart",
                "ecr:ListImages",
                "logs:CreateLogGroup",
                "ecr:PutImage",
                "iam:PassRole",
                "sagemaker:*",
                "ecr:BatchGetImage",
                "ecr:CompleteLayerUpload",
                "ecr:DescribeImages",
                "ecr:DescribeRepositories",
                "ecr:InitiateLayerUpload",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "*"
        }
    ]
}
```

14. Choose <b>Next:Tags</b> and add a tag, if you want to.
15. Choose <b>Next:Review</b> and add a name such as <b>AmazonSageMaker-ExecutionPolicy</b>.
16. Choose <b>Create Policy</b>.
17. Select <b>Roles</b> and search for your <b>role</b>.
18. Under the <b>Permissions</b> tab, click <b>Attach policies</b>.
19. Search for your newly created policy and select the check box next to it.
20. Choose <b>Attach policy</b>.    

### Docker images

First, we need to create an image and upload it in ECR for each one of the steps. To do that, execute the following commands in the terminal:

```bash
cd Sagemaker/project-name
source project-name_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name project-name_env --display-name "project-name"
bash build_and_push.sh processor hermione-processor
bash build_and_push.sh train hermione-train
bash build_and_push.sh inference hermione-inference
```

The bash command will access the Dockerfile in the folder, create the image and save it in ECR with the specified name

### Notebooks

To test the images in ECR, execute the following notebooks:
  
- project-name/src/ml/notebooks/1_Sagemaker_Processor.ipynb
- project-name/src/ml/notebooks/2_Sagemaker_Train.ipynb
- project-name/src/ml/notebooks/3_Sagemaker_Inference.ipynb

## Stepfunctions

We also create two Step Function state machines to execute the whole process. The first machine processes the training data and creates the model. And the second one processes the inference data and generates its prediction.

### Permitions

The Step Functions workflow requires an IAM role to interact with other services in AWS environment. To do that, follow these [AWS steps](https://github.com/aws/amazon-sagemaker-examples/blob/master/step-functions-data-science-sdk/step_functions_mlworkflow_processing/step_functions_mlworkflow_scikit_learn_data_processing_and_model_evaluation.ipynb):


1. Go to the [<b>IAM console</b>](https://console.aws.amazon.com/iam/).
2. Select <b>Roles</b> and then <b>Create role</b>.
3. Under <b>Choose the service that will use this role</b> select <b>Step Functions</b>.
4. Choose <b>Next</b> until you can enter a <b>Role name</b>.
5. Enter a name such as <b>AmazonSageMaker-StepFunctionsWorkflowExecutionRole</b> and then select <b>Create role</b>.
6. Search and click on the IAM Role you just created.
7. Click <b>Attach policies</b> and then select <b>CloudWatchEventsFullAccess</b>.
9. Click on <b>Attach Policy</b>


Next, create and attach another new policy to the role you created:

9. Click <b>Attach policies</b> again and then <b>Create policy</b>.
10. Enter the following in the <b>JSON</b> tab:


```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "events:PutTargets",
                "events:DescribeRule",
                "events:PutRule"
            ],
            "Resource": [
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForStepFunctionsExecutionRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule"
            ]
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "NOTEBOOK_ROLE_ARN",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        },
        {
            "Sid": "VisualEditor2",
            "Effect": "Allow",
            "Action": [
                "batch:DescribeJobs",
                "batch:SubmitJob",
                "batch:TerminateJob",
                "dynamodb:DeleteItem",
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "ecs:DescribeTasks",
                "ecs:RunTask",
                "ecs:StopTask",
                "glue:BatchStopJobRun",
                "glue:GetJobRun",
                "glue:GetJobRuns",
                "glue:StartJobRun",
                "lambda:InvokeFunction",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateHyperParameterTuningJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateProcessingJob",
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateTransformJob",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DescribeHyperParameterTuningJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:DescribeTransformJob",
                "sagemaker:ListProcessingJobs",
                "sagemaker:ListTags",
                "sagemaker:StopHyperParameterTuningJob",
                "sagemaker:StopProcessingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:StopTransformJob",
                "sagemaker:UpdateEndpoint",
                "sns:Publish",
                "sqs:SendMessage"
            ],
            "Resource": "*"
        }
    ]
}
```
    
11. Replace <b>NOTEBOOK_ROLE_ARN</b> with the ARN for your notebook that you used in the previous step in the above Sagemaker Permitions.
12. Choose <b>Review policy</b> and give the policy a name such as <b>AmazonSageMaker-StepFunctionsWorkflowExecutionPolicy</b>.
13. Choose <b>Create policy</b>.
14. Select <b>Roles</b> and search for your <b>AmazonSageMaker-StepFunctionsWorkflowExecutionRole</b> role.
15. Click <b>Attach policies</b>.
16. Search for your newly created <b>AmazonSageMaker-StepFunctionsWorkflowExecutionPolicy</b> policy and select the check box next to it.
17. Choose <b>Attach policy</b>.
18. Copy the <b>AmazonSageMaker-StepFunctionsWorkflowExecutionRole</b> Role ARN at the top of the Summary. You will use it in the next step.


### Notebooks

To create and test the Step Functions state machines, execute the following notebooks:

- project-name/src/ml/notebooks/4_Sagemaker_StepFunctions_Train.ipynb
- project-name/src/ml/notebooks/5_Sagemaker_StepFunctions_Inference.ipynb