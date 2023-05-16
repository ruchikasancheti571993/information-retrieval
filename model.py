from sentence_transformers import SentenceTransformer
from config import MODEL_PATH

old_model_name = "msmarco-distilbert-base-tas-b"
old_model = SentenceTransformer(old_model_name)

# new_model_name = "/content/notebooks/biencoder-arxiv"
new_model_name = MODEL_PATH
new_model = SentenceTransformer(new_model_name)

#We load the TAS-B model, a state-of-the-art model trained on MS MARCO
max_seq_length = 256
model_name = "msmarco-distilbert-base-tas-b"

org_model = SentenceTransformer(model_name)
org_model.max_seq_length = max_seq_length