import os
import tarfile


def extract_tar_gz(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar_fp:
        tar_fp.extractall(path=extract_path)


def repackage_model(model_artifact, source_artifact, tmp_dir, target_dir):
    tmp_code_dir = os.path.join(tmp_dir, 'code')

    extract_tar_gz(model_artifact, tmp_dir)
    extract_tar_gz(source_artifact, tmp_code_dir)

    with tarfile.open(os.path.join(target_dir, 'model.tar.gz'), 'w:gz') as tar:
        tar.add(tmp_dir, arcname='.')


if __name__ == "__main__":
    model_artifact_path = '/opt/ml/processing/model/model.tar.gz'
    source_artifact_path = '/opt/ml/processing/source/source.tar.gz'
    output_dir = '/opt/ml/processing/output'
    temp_dir = '/opt/ml/processing/temp'

    repackage_model(model_artifact_path, source_artifact_path, temp_dir, output_dir)
