
import os
import pinecone
import tqdm

class PineconeWrapper():
    def __init__(self, model, top_k=1, dimensions=768):
        pinecone.init(api_key=os.environ["PINECONE_KEY"], environment='us-west1-gcp')
        # create new mining index if does not exist
        if 'negative-mines-petal' not in pinecone.list_indexes():
            pinecone.create_index(
                'negative-mines-petal', dimension=dimensions,
                metric='dotproduct', pods=1, pod_type='p1'  # limit of pods=1 for free plan (more pods == faster mining)
            )
        # connect
        self.index = pinecone.Index('negative-mines-petal')
        print("Index Stats: ", self.index)

        self.batch_size = 16
        self.dimension = dimensions
        self.model = model
        self.top_k = top_k


    def upload_pinecone(self, haystack_docs, namespace):
        docs = [haystack_docs[i]['content'] for i in range(len(haystack_docs))]

        # doc_dir = [haystack_docs[i]['meta']['doc_dir'] for i in range(len(haystack_docs))]

        print("\nUPLOAD PINECONE START!*******!")
        docs_emb = self.model.encode(docs, convert_to_tensor=True, show_progress_bar=True)
        print(f"Document Embeddings Shape : {docs_emb.shape}")

        index_data = self.index.describe_index_stats()
        print(f"Index Data before Adding : {index_data}")
        totalVectorCount = int(index_data['total_vector_count'])

        for i in tqdm.tqdm(range(0, len(docs_emb), self.batch_size)):
            i_end = min(i+self.batch_size, len(docs_emb))
            batch_emb = docs_emb[i:i_end, :].tolist()
            # batch_data = docs[i:i_end]
            
            # batch_metadata = [{"text": batch_data[i]} for i in range(0, len(batch_data))]
            batch_metadata = [{"text": haystack_docs[j]['content'], 
                               "doc_dir": haystack_docs[j]['meta']['doc_dir']} 
                              for j in range(i, i_end)]

            batch_ids = [str(x+totalVectorCount) for x in range(i, i_end)]
            # print(f"Batch ID : {batch_ids}, Batch MetaData : {batch_metadata}")
            # print(f"Batch ID : {batch_ids}, Batch Embeddings : {batch_emb}")
            # print(f"Batch ID : {batch_ids}, Batch Data : {batch_data}")
            # upload to index
            upload_vectors = list(zip(batch_ids, batch_emb, batch_metadata))
            print(f"\nBatch Upload Vectors : {upload_vectors}\n")
            self.index.upsert(vectors=upload_vectors, namespace=namespace)
        
        index_data = self.index.describe_index_stats()
        print(f"Index Data after Adding : {index_data}")
        print("\nUPLOAD PINECONE END!*******!")

    def delete_vectors(self, namespace=None):
        if not namespace:
            index_data = self.index.describe_index_stats()
            print(f"Index Data before deleting : {index_data}")

            to_delete = []
            for i in range(12263, 12294):
                to_delete.append(str(i))
            print(f"Index to Delete : {to_delete}")
            self.index.delete(ids = to_delete)

            index_data = self.index.describe_index_stats()
            print(f"Index Data after deleting : {index_data}")
        else:
            index_data = self.index.describe_index_stats()
            print(f"Index Data before deleting : {index_data}")

            self.index.delete(delete_all=True, namespace=namespace)

            index_data = self.index.describe_index_stats()
            print(f"Index Data after deleting : {index_data}")

    def query_pinecone(self, query, namespace):
        query_emb = self.model.encode(query).tolist()
        # print(query_emb.shape)
        # print(query_emb)
        
        # res = index.query([query_emb], top_k = 10)
        if namespace:
            res = self.index.query([query_emb], top_k = self.top_k, namespace=namespace, include_metadata=True)
        else:
            res = self.index.query([query_emb], top_k = self.top_k)
        print(f"Pinecone Results : {res}")
        
        # ids = [match.id for match in res['results'][0]['matches']]
        # scores = [match.score for match in res['results'][0]['matches']]
        
        # print(ids)
        # print(scores)
        answers = []
        
        for match in res['matches']:
            vector_id = int(match.id)
            score = match.score
            # text = corpus[vector_id]
            try:
                text = match['metadata']['text']
                doc_dir = match['metadata']['doc_dir']
            except:
                text = "NA"
                doc_dir = "NA"
                
            result_dict = {
                "content" : text,
                "context-type" : "text",
                "meta":{
                    "id" : vector_id,
                    "score": score,
                    "doc_dir" : doc_dir
                }
            }
            answers.append(result_dict)
            # print(f"Score : {score}, ID : {vector_id}, TEXT : {text}")
        
        return answers