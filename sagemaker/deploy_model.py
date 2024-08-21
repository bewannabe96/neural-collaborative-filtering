import time

import boto3


def handler(event, context):
    model_name = event['model_name']
    role = event['role']
    model_package_arn = event['model_package_arn']
    instance_type = event['instance_type']

    unq_model_name = '{}-{}'.format(model_name, time.strftime('%Y%m%d-%H%M%S'))

    sagemaker_client = boto3.client('sagemaker')

    delete_models = []
    delete_endpoint_configs = []

    response = sagemaker_client.list_models(NameContains=model_name)
    for model in response['Models']:
        delete_models.append(model['ModelName'])

    response = sagemaker_client.list_endpoint_configs(NameContains=model_name)
    for endpoint_config in response['EndpointConfigs']:
        delete_endpoint_configs.append(endpoint_config['EndpointConfigName'])

    sagemaker_client.create_model(
        ModelName=unq_model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'Mode': 'SingleModel',
            'ModelPackageName': model_package_arn,
        }
    )

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=unq_model_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'InitialVariantWeight': 1,
                'ModelName': unq_model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': 1,
            }
        ],
    )

    sagemaker_client.update_endpoint(
        EndpointName=model_name,
        EndpointConfigName=unq_model_name,
    )

    for name in delete_models:
        sagemaker_client.delete_model(ModelName=name)

    for name in delete_endpoint_configs:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=name)
