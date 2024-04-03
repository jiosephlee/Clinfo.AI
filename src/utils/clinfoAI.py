import sys
import os 
sys.path.append('..')
from  pathlib  import  Path
from  utils.pubmed_utils     import Neural_Retriever_PubMed
from  utils.semantic_utils   import Neural_Retriever_Semantic_Scholar
from utils.bm25              import bm25_ranked
from openai import OpenAI
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ClinfoAI:
    def __init__(self,openai_key, email,engine="SemanticScholar",verbose=False) -> None:
        self.engine             = engine
        self.email              = email
        self.openai_key         = openai_key
        self.verbose            = verbose
        self.architecture_path  = self.init_engine()


    def init_engine(self):
        if self.engine  == "PubMed":
            ARCHITECTURE_PATH = Path('../prompts/PubMed/Architecture_1/master.json')
            self.NEURAL_RETRIVER   = Neural_Retriever_PubMed(architecture_path=ARCHITECTURE_PATH ,verbose=False,debug=True,open_ai_key=self.openai_key,email=self.email)
            print("PubMed Retriever Initialized")

        elif self.engine  == "SemanticScholar":
            ARCHITECTURE_PATH    = Path('../prompts/SemanticScholar/Architecture_1/master.json')
            self.NEURAL_RETRIVER      = Neural_Retriever_Semantic_Scholar(architecture_path=ARCHITECTURE_PATH ,verbose=True,debug=True,open_ai_key=self.openai_key,email=self.email)
        else:
            raise Exception("Invalid Engine")

        ARCHITECTURE_PATH_STR = str(ARCHITECTURE_PATH)
        return   ARCHITECTURE_PATH_STR 
    

    def retrive_articles(self,question,restriction_date = None, ignore=None):
        try:
            if self.engine  == "PubMed":
                queries, article_ids = self.NEURAL_RETRIVER.search_pubmed(question             = question,
                                                                            num_results        = 16,
                                                                            num_query_attempts = 3,
                                                                            restriction_date   = restriction_date) 
                

            elif self.engine  == "SemanticScholar":
                query        = self.NEURAL_RETRIVER.generate_semantic_query(question=question)
                article_ids  = [1,2,3]
                queries      = [query]
        except: 
            print(f"Internal Service Error, {self.engine } might be down ")
         
       
        

        if ignore != None:
            try:
                print("Article dropped")
                article_ids.remove(ignore)
            except:
                pass




        if (not article_ids) or (not queries) or (len(article_ids) == 0) or (len(queries) == 0):
            print(f"Sorry, we weren't able to find any articles in {self.engine} relevant to your question. Please try again.")
            return
    
        try:
            if self.engine == "PubMed":
                articles = self.NEURAL_RETRIVER.fetch_article_data(article_ids)
            
            elif self.engine == "SemanticScholar":
                articles    = self.NEURAL_RETRIVER.search_semantic_scholar(query,limit=50,threshold = 10,minimum_return=5,verbose=True)
                article_ids = articles

            if  self.verbose:
                print(f'Retrieved {len(articles)} articles. Identifying the relevant ones and summarizing them (this may take a minute)')
            

        except:
            print('error',f"Articles could not be fetched from {self.engine}")
        

        
        if len(articles) ==0:
            print(f"Articles could not be fetched from {self.engine}, 0")
           
        return articles,queries
    

    def summarize_relevant(self,articles,question):
        article_summaries,irrelevant_articles = self.NEURAL_RETRIVER.summarize_each_article(articles, question,prompt_dict={"type":"automatic"})
        return   article_summaries,irrelevant_articles 
    


    def synthesis_task(self,article_summaries, question,USE_BM25=False,with_url=True ):
        if USE_BM25:
            if len(article_summaries) > 21:
                print("Using BM25 to rank articles")
                corpus            = [article['abstract'] for article in article_summaries]
                article_summaries = bm25_ranked(list_to_oganize= article_summaries,corpus =  corpus,query = question,n = 20)

        synthesis = self.NEURAL_RETRIVER.synthesize_all_articles(article_summaries, question, prompt_dict={"type":"automatic"} ,with_url=with_url)
        return synthesis
    


    def forward(self,question,restriction_date = None, ignore=None,return_articles=True):  
        try:
            articles,queries                              = self.retrive_articles(question,restriction_date , ignore)
            article_summaries,irrelevant_articles  = self.summarize_relevant(articles=articles,question=question)
            synthesis                              = self.synthesis_task(article_summaries, question)
        except:
            synthesis = "Internal Error"
        
        if return_articles:
            return {"synthesis": synthesis , "article_summaries": article_summaries, "irrelevant_articles" : irrelevant_articles, "queries" : queries}
        
        return synthesis 

    
class ClinfoAIForQA:
    def __init__(self,openai_key, email,engine="PubMed",verbose=False, temperature=0.0) -> None:
        self.engine             = engine
        self.email              = email
        self.openai_key         = openai_key
        self.verbose            = verbose
        self.temperature = temperature
        self.client = OpenAI(api_key=openai_key)
        self.architecture_path  = self.init_engine()

    def init_engine(self):
        if self.engine  == "PubMed":
            ARCHITECTURE_PATH = Path('../prompts/PubMed/Architecture_1/master.json')
            self.NEURAL_RETRIVER   = Neural_Retriever_PubMed(architecture_path=ARCHITECTURE_PATH , temperature=0,verbose=False,debug=False,open_ai_key=self.openai_key,email=self.email)
            print("PubMed Retriever Initialized")

        elif self.engine  == "Ada":
            ARCHITECTURE_PATH = Path('../prompts/PubMed/Architecture_1/master.json')
            self.NEURAL_RETRIVER   = Neural_Retriever_PubMed(architecture_path=ARCHITECTURE_PATH , temperature=0,verbose=False,debug=False,open_ai_key=self.openai_key,email=self.email)
            print("PubMed Retriever Initialized")
            with open('../qa/document_embeddings_txt_size128_openaiAPI.pkl','rb') as f2:
                self.document_embeddings = pickle.load(f2)
                self.document_embeddings_array = np.array(self.document_embeddings["embeddings"])
            print("Embeddings loaded")
        else:
            raise Exception("Invalid Engine")

        ARCHITECTURE_PATH_STR = str(ARCHITECTURE_PATH)
        return   ARCHITECTURE_PATH_STR 
    
    def get_relevant(self,articles,question):
        relevant_articles = self.NEURAL_RETRIVER.summarize_each_article(articles, question)
        return   relevant_articles
    
    def summarize_relevant(self,articles,question):
        article_summaries,irrelevant_articles = self.NEURAL_RETRIVER.summarize_each_article(articles, question)
        return   article_summaries,irrelevant_articles 
    
    def summarize_relevant_texts(self,texts,question):
        article_texts,irrelevant_texts = self.NEURAL_RETRIVER.summarize_each_text(texts, question)
        return   article_texts,irrelevant_texts
    
    def retrive_articles_vector(self, question_embedding, k = 5):
        question_embedding = np.array(question_embedding).reshape(1, -1)
        print(question_embedding.shape)
        # Calculate cosine similarity
        cos_similarities = cosine_similarity(question_embedding, self.document_embeddings_array)
        # Find the index of the highest similarity
        most_similar_indices = np.argpartition(cos_similarities, -k)[0][-k:]
        # Retrieve the corresponding document
        most_similar_documents = [self.document_embeddings["document"][i] for i in most_similar_indices]

        print("Most similar documents:", most_similar_documents)
        return most_similar_documents
    
    # EMBEDDING_MODEL = "text-embedding-ada-002"  #"text-embedding-3-small"
    # client = OpenAI()
    # def strings_ranked_by_relatedness(
    #     query: str,
    #     df: pd.DataFrame,
    #     relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    #     top_n: int = 100
    # ) -> tuple[list[str], list[float]]:
    #     """Returns a list of strings and relatednesses, sorted from most related to least."""
    #     query_embedding_response = client.embeddings.create(
    #         model=EMBEDDING_MODEL,
    #         input=[query.replace('\n', ' ')]
    #     )
    #     query_embedding = query_embedding_response.data[0].embedding
    #     strings_and_relatednesses = [
    #         (row["document"], relatedness_fn(query_embedding, row["embeddings"]))
    #         for i, row in df.iterrows()
    #     ]
    #     strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    #     strings, relatednesses = zip(*strings_and_relatednesses)
    #     return strings[:top_n], relatednesses[:top_n]
    # strings, relatednesses = strings_ranked_by_relatedness(question, AD_literature_chunks, top_n=3)
    
    def retrive_articles(self,question,restriction_date = None, ignore=None, years_back = 20):
        count = 0
        try:
            if self.engine  == "PubMed":
                queries, article_ids = self.NEURAL_RETRIVER.search_pubmed(question             = question,
                                                                            num_results        = 5,
                                                                            num_query_attempts = 3,
                                                                            restriction_date   = restriction_date,
                                                                            years_back = years_back)
            while(len(article_ids) <= 0 and count < 12):
                print("let's try again")
                count += 1
                queries, article_ids = self.NEURAL_RETRIVER.search_pubmed(question             = question,
                                                                    num_results        = 5,
                                                                    num_query_attempts = 3,
                                                                    restriction_date   = restriction_date,
                                                                    years_back = years_back,
                                                                    retry = True)
        except: 
            print(f"Internal Service Error, {self.engine } might be down ")

        if ignore != None:
            try:
                print("Article dropped")
                article_ids.remove(ignore)
            except:
                pass



        if (not article_ids) or (not queries) or (len(article_ids) == 0) or (len(queries) == 0):
            print(f"Sorry, we weren't able to find any articles in {self.engine} relevant to your question. Please try again.")
            return
    
        try:
            if self.engine == "PubMed":
                articles = self.NEURAL_RETRIVER.fetch_article_data(article_ids)
            if  self.verbose:
                print(f'Retrieved {len(articles)} articles. Identifying the relevant ones and summarizing them (this may take a minute)')
            

        except:
            print('error',f"Articles could not be fetched from {self.engine}")
        

        
        if len(articles) ==0:
            print(f"Articles could not be fetched from {self.engine}, 0")
           
        return articles,queries
    
    def answer_q(self,articles, question,num_articles=5, USE_BM25=False,with_url=True ):
        if USE_BM25:
            if len(articles) > num_articles:
                print("Using BM25 to rank articles")
                corpus            = [article['abstract'] for article in articles]
                articles = bm25_ranked(list_to_oganize= articles,corpus =  corpus,query = question,n = num_articles)

        synthesis = self.NEURAL_RETRIVER.answer_question_with_articles(articles, question, prompt_dict={"type":"automatic"} ,with_url=with_url)
        return synthesis

    def request_api_gpt(self,prompt):
        GPT_MODEL =  "gpt-3.5-turbo" # 'gpt-4-1106-preview'
        response = self.client.chat.completions.create(
            messages = [
                {"role": "user", "content" : prompt}
            ],
            model=GPT_MODEL,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def pubmed_forward(self,question,restriction_date = None, ignore=None,num_articles=3, years_back = 20):  
        print('-------retrieving articles-------')
        articles,queries                              = self.retrive_articles(question,restriction_date , ignore, years_back = years_back)
        print('----------summarizing relevant articles---------')
        article_summaries,irrelevant_articles  = self.summarize_relevant(articles=articles,question=question)
        print('number of relevant articles: ',len(article_summaries))
        print('------synthesizing------')
        prompt                              = self.answer_q(article_summaries, question, num_articles=num_articles,USE_BM25=True)
        print('prompt: ',prompt)
        answer = self.request_api_gpt(prompt[0].dict()['content'] + '\n' + prompt[1].dict()['content'])
        return answer

    def forward(self,question,num_articles=3):  
        print('-------encoding question------')
        question_embedding = self.client.embeddings.create(
            input=[question],
            model="text-embedding-ada-002"
            ).data[0].embedding

        print('-------retrieving articles-------')

        articles                             = self.retrive_articles_vector(question_embedding, k = 10)
        article_summaries, irrelevant_articles                    = self.summarize_relevant_texts(articles, question)
        print('------answering------')
        prompt                           = self.answer_q(article_summaries, question, USE_BM25=True, num_articles=num_articles)
        print('prompt: ',prompt)
        answer = self.request_api_gpt(prompt[0].dict()['content'] + '\n' + prompt[1].dict()['content'])
        return answer