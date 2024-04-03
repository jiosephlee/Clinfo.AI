import json
import os
from sentence_transformers import SentenceTransformer
import argparse

embedding_models = {
    'sentence_transformer' : SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sentence_transformer')
    parser.add_argument('--gpu', type=str, default= 'cpu')
    args = parser.parse_args()

    model = embedding_models[args.model]
    
    if args.gpu == 'cpu':
        model.to("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
        model.to("cuda")

    years = [2011,2012,2013,2014,2015,2016,2017, 2018, 2019, 2020]

    document = []
    for year in years:
        with open(os.path.join('..', '..', 'by_year', '{}.pubtator'.format(year))) as f:
            for line in f.readlines():
                line = line.strip()
                if '|a|' in line:
                    abstract = line.split('|a|')[1]
                    abstract_list = abstract.split('.')
                    chunk = ''
                    for item in abstract_list:
                        if len(chunk+'. '+item) <= 256:
                            chunk = chunk+'. '+item
                        else:
                            document.append(chunk)
                            chunk = item

    embeddings = model.encode(document, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    keyword_emb_dict = {
        "document": document,
        "embeddings": embeddings,
    }
    import pickle
    with open("document_embeddings_sentence.pkl", "wb") as f:
        pickle.dump(keyword_emb_dict, f)

if __name__ == '__main__':
    main()