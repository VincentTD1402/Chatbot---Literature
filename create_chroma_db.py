from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer,util
import pandas as pd
import numpy as np
from googlesearch import search

class LocalChromaDB():
    client = PersistentClient(path="vanhoc_db")
    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    )

    def __init__(self):
        pass

    def check_db(self):
        '''
        Check whether database exists
        '''
        if self.client.list_collections():
            print('client exist!')
            return True
        else:
            print("client doesn't exist")
            return False

    def extract_question_data_adapter(self,
            link_data: str,
            is_save: bool=True
        ):
        '''
        Extract neccessary fields of questions to create chroma database
        Args:
            - link_data (str): link to excel file
            - is_save (bool): whether to save preprocessed data into csv file
        '''
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        index = 0
        dataset = pd.read_excel(link_data)
        dataset['Tonghop'] = ""
        self.data = dataset["Câu hỏi"]
        if link_data == "collections/cauhoivanhoccap2(hieuchinhlan1) 1.xlsx":
            for index, row in dataset.iterrows():
                dataset.loc[index, 'Tonghop'] = "Câu hỏi: {} /n trả lời: {}".format(row["Câu hỏi"], row["Trả lời"])
                documents.append(row['Câu hỏi'])
                embedding = self.model.encode(row['Câu hỏi']).tolist()
                embeddings.append(embedding)
                metadatas.append({'source': row['Tác phẩm'],'trả lời': row["Trả lời"]})
                ids.append(str(index + 1))
            if is_save:
                dataset.to_excel("collections/DataCauHoi2.xlsx")
        elif link_data == "collections/tomtatvanhoccap2(dahieuchinhlan1).xlsx":
            for index, row in dataset.iterrows():
                dataset.loc[index, 'Tonghop'] = "Câu hỏi: {} /n Tóm tắt: {}".format(row["Câu hỏi"], row["Tóm tắt"])
                documents.append(row['Câu hỏi'])
                embedding = self.model.encode(row['Câu hỏi']).tolist()
                embeddings.append(embedding)
                metadatas.append({'source': row['Tác phẩm'],'Tóm tắt': row["Tóm tắt"]})
                ids.append(str(index + 1))
            if is_save:
                dataset.to_excel("collections/DataTomTat2.xlsx")
        return documents, embeddings, metadatas, ids
    
    def create_db_summary(self, 
            link_data: str, 
            name_collection: str
        ):
        '''
        Create database for Vietnam Literary Story summary
        Args:
            - link_data (str): link of excel file including the neccessary information to save in collection.
            - name_collection (str): name of chroma database collection.
        '''
        documents, embeddings, metadatas, ids= self.extract_question_data_adapter(link_data)
        literature_collection = self.client.create_collection(name=name_collection,
                                                    metadata={"hnsw:space": "cosine"},
                                                    embedding_function=self.sentence_transformer_ef)
        literature_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("Data has been added in collection {}".format(name_collection))

    def auto_search(self,query, num_results):
        '''
        Search using google search
        Args:
            query (str) : The input of question 
            num_result (int): Number of result searching
        '''
        search_results = search(query, num_results=num_results)
        return list(search_results)
    
    def find_summary(self,
            name_collection: str,
            question: str,
            num_of_answer:int=1,
            story_name: str=None
        ):
        '''
        Finding summary from database
        Args:
            - name_collection (str): Name of collection
            - story_name (str): The story name to limit the scope of searching similar queries
            - question (str): The input question
            - num_of_answer (int): Number of return similar queries 
        '''
        literature_collection = self.client.get_collection(
            name=name_collection,
            embedding_function=self.sentence_transformer_ef)
        
        results = literature_collection.query(
            query_texts=[question],
            n_results=num_of_answer,
            where_document={"$contains": story_name}
        )

        # for i in results['metadatas'][0]:
            # print('Relevant summary: ', i)
        return [item['Tóm tắt'] for item in results['metadatas'][0]]

    def find_sim_queries(self,
            name_collection: str,
            story_name: str, 
            question: str,
            num_of_answer:int=1
        ):
        '''
        Finding similarity queries
        Args:
            - name_collection (str): Name of collection
            - story_name (str): The story name to limit the scope of searching similar queries
            - question (str): The input question
            - num_of_answer (int): Number of return similar queries 
        '''
        literature_collection = self.client.get_collection(
            name=name_collection,
            embedding_function=self.sentence_transformer_ef)
        
        results = literature_collection.query(
            query_texts=[question],
            n_results=num_of_answer,
            where_document={"$contains": story_name}
        )

        for i in results['documents'][0]:
            print('Relevant Query: ', i)
        return results['documents']
    
    def find_score_answer(self, query):
        embeddings_search_cauhoi = np.load("embedding_search/embeddings_search_cauhoi.npy")
        question_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, embeddings_search_cauhoi, top_k=1)
        hits = hits[0]
        for hit in hits:
            temp = hit['score']
            return temp
        
    def find_sim_answer(self,
            name_collection: str,
            story_name: str, 
            question: str,
            num_of_answer:int=1
        ):
        '''
        Finding similarity queries
        Args:
            - name_collection (str): Name of collection
            - story_name (str): The story name to limit the scope of searching similar queries
            - question (str): The input question
            - num_of_answer (int): Number of return similar queries 
        '''
        literature_collection = self.client.get_collection(
            name=name_collection,
            embedding_function=self.sentence_transformer_ef)
        
        results = literature_collection.query(
            query_texts=[question],
            n_results=num_of_answer,
            where_document={"$contains": story_name}
        )

        sim_score = self.find_score_answer(query=question)

        if sim_score >= 0.9:
            relevant_answers = results['metadatas'][0]
            if relevant_answers:
                first_answer = relevant_answers[0]
                answer = first_answer.get('trả lời', 'N/A')
                print('Relevant Answer (trả lời):', answer)
                return answer
        elif 0.8 <= sim_score < 0.9:
            pre_announce = "Rất xin lỗi, dường như câu hỏi của bạn chưa thực sự rõ ràng hoặc câu hỏi vượt quá phạm vi của tác phẩm '{}'. Có phải bạn muốn hỏi một trong các câu hỏi sau đây ?".format(story_name)
            similar_questions = self.find_sim_queries(
                name_collection=name_collection,
                story_name=story_name,
                question=question,
                num_of_answer=5
            )
            similar_question_list =  [f"<br>{idx + 1}) {similar_question}" for idx, similar_question in enumerate(similar_questions[0])]
            similar_question_list = ''.join(similar_question_list)
            similar_question_list = pre_announce + str(similar_question_list)
            none_ans = '[]'
            try: 
                none_ans = self.auto_search(query = f"văn học cấp 2 " + story_name , num_results = 4)
                none_ans = str(none_ans)
                similar_question_list + '\n Sau đây là một số địa chỉ website bên ngoài liên quan tới câu hỏi của bạn: ' + none_ans
            except Exception as e:
                print("Cannot find nay external websites")
            return similar_question_list
        else:
            return
        
        # return None
        # else:
        #     none_ans = '[]'
        #     pre_announce = """
        #     <span> Xin phép được giới thiệu, tôi là Vietnam Literary Assistant, một hệ thống trợ lý văn học Việt Nam do công ty Neurond AI phát triển. Tôi được huấn luyện với mục đích cung cấp thông tin về văn học Việt Nam, bài thơ, tiểu thuyết, tác giả, và nhiều nội dung khác. Tôi cố gắng cung cấp câu trả lời tốt nhất dựa trên kiến thức có sẵn trong dữ liệu đào tạo của tôi. </span>
        #     <br>
        #     <br>
        #     <span> <b>Tuy nhiên, có thể có những câu hỏi hoặc thông tin mới mà tôi chưa từng gặp trước, hoặc nằm ngoài phạm vi văn học, do đó tôi có thể không trả lời được một số câu hỏi.</b> Điều này có thể xảy ra khi câu hỏi của bạn rơi vào một lĩnh vực hoặc chủ đề mà tôi chưa có đủ thông tin hoặc hiểu biết. </span>
        #     <br>
        #     <br>
        #     <span> Để cải thiện chất lượng câu trả lời, tôi rất mong nhận được phản hồi từ người dùng. Nếu bạn có thông tin bổ sung hoặc câu trả lời chính xác cho câu hỏi mà tôi chưa trả lời được, xin vui lòng chia sẻ nó với chúng tôi trong phần phản hồi. Điều này sẽ giúp tôi cải thiện và cung cấp dịch vụ tốt hơn trong tương lai. </span>        
        #     <br>
        #     """
        #     try:
        #         none_ans = self.auto_search(query = f"văn học cấp 2 " + story_name , num_results = 4)
        #         none_ans = ['<a>href={link}<a>'.format(link) for link in none_ans]
        #         none_ans = ''.join(none_ans)
        #         none_ans = pre_announce + """
        #         <br>
        #         <br>
        #         <span> Chúng tôi xin lỗi vì không thể mang lại trải nghiệm tốt nhất. Để bù đắp, chúng tôi tìm thấy số địa chỉ website bên ngoài liên quan tới câu hỏi của bạn: </span>
        #         """ + none_ans
        #     except Exception as e:
        #         none_ans = pre_announce
        #         # TODO: Feedback for each answer
        #     return none_ans

if __name__ == "__main__":
    # 1. Test creating chromadb
    link_data="collections/cauhoivanhoccap2(hieuchinhlan1) 1.xlsx"
    name_data = "cauhoivanhoccap2-vn-supsimcse"
    chromadb = LocalChromaDB()
    chromadb.create_db_summary(link_data, name_data)
    chromadb.check_db()
    # # 2. Test query question
    # similar_queries = chromadb.find_sim_queries(
    #     name_collection='cauhoivanhoccap2-vn-supsimcse',
    #     story_name='Ông lão đánh cá và con cá vàng',
    #     question='Ông lão đánh cá gặp chú cá vàng trong hoàn cảnh nào?',
    #     num_of_answer=3
    # )

    # # 3. Test the answer
    # similar_queries = chromadb.find_sim_answer(
    #     name_collection='cauhoivanhoccap2-vn-supsimcse',s
    #     story_name='Ông lão đánh cá và con cá vàng',
    #     question='ý nghĩa',
    #     num_of_answer=1
    # )