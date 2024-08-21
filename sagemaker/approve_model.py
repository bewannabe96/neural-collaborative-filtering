import boto3


def get_model_package_arn(sagemaker_client, model_package_group_name, model_package_version):
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10,
    )

    model_package_arn = None
    for model_package in response['ModelPackageSummaryList']:
        if model_package['ModelPackageVersion'] == model_package_version:
            model_package_arn = model_package['ModelPackageArn']
            break

    return model_package_arn


def get_latest_model_package_arn(sagemaker_client, model_package_group_name):
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1,
    )

    if len(response['ModelPackageSummaryList']) == 1:
        return response['ModelPackageSummaryList'][0]['ModelPackageArn']
    else:
        return None


def handler(event, context):
    try:
        sagemaker_client = boto3.client('sagemaker')

        model_package_group_name = event['model_package_group_name']
        print('Model Package Group Name: {}'.format(model_package_group_name))

        # model_package_version = event['model_package_version']
        # print('Model Package Version: {}'.format(model_package_version))
        # model_package_arn = get_model_package_arn(sagemaker_client, model_package_group_name, model_package_version)
        model_package_arn = get_latest_model_package_arn(sagemaker_client, model_package_group_name)

        if model_package_arn is None:
            raise Exception('Model Package Version Not Found')
        print('Model Package ARN: {}'.format(model_package_arn))

        sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved',
        )

        return {'statusCode': 200, 'modelPackageArn': model_package_arn}

    except Exception as e:
        print(e)
        return {'statusCode': 500}
