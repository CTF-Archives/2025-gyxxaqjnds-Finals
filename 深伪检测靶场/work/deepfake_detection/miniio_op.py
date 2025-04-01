import os
from minio import Minio
from minio.error import S3Error


def get_bool_env(var_name, default=False):
    """  
    从环境变量中获取布尔值。  
  
    参数:  
    var_name (str): 环境变量的名称。  
    default (bool): 如果环境变量不存在或无法转换为布尔值，则返回此默认值。  
  
    返回:  
    bool: 环境变量的布尔值。  
    """  
    value = os.getenv(var_name, str(default).lower())
    if value.lower() in ['true', '1', 'yes', 'y']:
        return True
    return False


def get_minio_client():
    minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio.secsmarts.com')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'Admin@2023')
    minio_secure = get_bool_env('MINIO_SECURE', False)
    print(f'minio_endpoint:{minio_endpoint}')
    # 连接到MinIO服务器
    minio_client = Minio(minio_endpoint,
                         access_key=minio_access_key,
                         secret_key=minio_secret_key,
                         secure=minio_secure)
    return minio_client


def upload(file_to_upload, minio_bucket_name, objectname):

    

    try:
        # 连接到MinIO服务器
        get_minio_client()
        if not minio_client.bucket_exists(minio_bucket_name):
            minio_client.make_bucket(minio_bucket_name)
        print(minio_bucket_name, objectname, file_to_upload)

      
        minio_client.fput_object(minio_bucket_name, objectname, file_to_upload)
       
    except S3Error as e:
        print(f'Error: {e}')


def check_and_download(model_name, cache_path):
    if os.path.exists(cache_path):
        return

    # 连接到MinIO服务器
    minio_client = get_minio_client()

    bucket = "deepfake-detectors"
    object_name = "%s.pth" % model_name


    minio_client.fget_object(bucket, object_name, cache_path)

def check_and_download_audio_models(model_name, cache_path):
    if os.path.exists(cache_path):
        return

    # 连接到MinIO服务器
    minio_client = get_minio_client()

    bucket = "deepfake-detectors"
    object_name = "%s.pt" % model_name

    minio_client.fget_object(bucket, object_name, cache_path)

def check_and_download_clip_basemodel(cache_path):
    if os.path.exists(cache_path):
        return
    

    # 连接到MinIO服务器
    minio_client = get_minio_client()


    bucket = "deepfake-detectors"
    object_name = "clip-vit-base-patch16.zip"


    minio_client.fget_object(bucket, object_name, cache_path+".zip")
    os.system("unzip -d %s %s.zip" % (cache_path, cache_path))


if __name__ == "__main__":
    print('-')
    # upload("/home/ubuntu1/SCNN.pt", "deepfake-detectors", "SCNN.pt")


