import generate_train
from utils import upload_pipeline
import sys

resp = upload_pipeline(sys.argv[2], getattr(generate_train, sys.argv[1]))
print(resp)