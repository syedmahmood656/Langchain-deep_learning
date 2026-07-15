import os
from dotenv import load_dotenv

load_dotenv()

print("HF Token:", os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
print("Alternative HF Token:", os.environ.get("HF_TOKEN"))