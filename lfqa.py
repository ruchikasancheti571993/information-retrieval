from haystack import Document
# from haystack.generator.transformers import Seq2SeqGenerator
from haystack.nodes import Seq2SeqGenerator
from haystack.utils import print_answers
from pathlib import Path
import os
import torch
import pandas as pd
from pinecone_util import PineconeWrapper
from model import new_model,old_model


def get_doc(all_doc_dir):
    threshold = 300
    doc_results = []

    print(f"Documents Found : {os.listdir(all_doc_dir)}")

    for document in os.listdir(all_doc_dir):
        doc_dir = os.path.join(all_doc_dir, document)
        print(f"Processing Document : {doc_dir}")
        text = Path(doc_dir).read_text().replace("\n", " ")
        text = text.replace('"', "")
        out = []
        
        for chunk in text.split('. '):
            if out and len(chunk)+len(out[-1]) < threshold:
                out[-1] += ' '+chunk+'.'
            else:
                out.append(chunk+'.')

        
        for doc in out:
            result_dict = {
                    "content" : doc,
                    "context-type" : "text",
                    "meta":{
                        "id" : None,
                        "score": 0,
                        "doc_dir" : document
                    }
                }
            doc_results.append(result_dict)

    return doc_results

def get_answers_wrapper(query, doc_dir, model, namespace, top_k):
    doc_results = get_doc(doc_dir)
    # print(f"DOC RESULTS : {doc_results}")

    pineconeWrapper = PineconeWrapper(model, top_k=top_k, dimensions = 768)

    pineconeWrapper.upload_pinecone(doc_results, namespace)

    biencoder_results = pineconeWrapper.query_pinecone(query, namespace)
    document_store = []

    data = {}
    data['query'] = query
    
    for idx, doc in enumerate(biencoder_results):
        i = str(idx)
        data["top_"+i+"_content"] = doc['content']
        data["top_"+i+"_dotscore"] = doc['meta']['score']
        data["top_"+i+"_document"] = doc['meta']['doc_dir']
        document_store.append(Document(doc['content']))

    print(f"All Documents : {document_store}")

    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")

    result = generator.predict(
        query=query,
        documents = document_store,
        top_k=1
    )

    print_answers(result, details="minimum")
    
    answers = result['answers']
    for answer in answers:
        final_ans = answer.answer
        ans_score = answer.score
        break
    
    data['answer'] = final_ans
    data['score'] = ans_score
    # return final_ans
    return data

def run_inference_refactored(query, doc_dir, model, top_k=5):
    pineconeWrapper = PineconeWrapper(model, top_k=top_k)
    pineconeWrapper.delete_vectors(namespace='artemis')
    data = get_answers_wrapper(query, doc_dir, model, namespace='artemis', top_k=top_k)
    return data

def get_csv(queries, all_doc_dir):
    all_data = []
    for query in queries:
        answers_new = run_inference_refactored(query, all_doc_dir, new_model, top_k=3)
        answers_new['model_type'] = 'trained'
        all_data.append(answers_new)

        answers_old = run_inference_refactored(query, all_doc_dir, old_model, top_k=3)
        answers_old['model_type'] = 'pretrained'
        all_data.append(answers_old)
    
    df = pd.DataFrame(all_data)
    return df