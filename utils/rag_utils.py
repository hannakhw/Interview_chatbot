from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import TokenTextSplitter
from operator import itemgetter
from dotenv import load_dotenv
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
env_path = os.path.join(root_dir, 'openai_key.env')

# .env 파일 로드
load_dotenv(env_path)
api_key = os.getenv('OPENAI_API_KEY')

def create_and_save_vectorstore():
    """벡터 스토어 생성 및 저장"""
    try:
        loader = DirectoryLoader("/Users/alookso/Desktop/2025 취준/hw_chatbot_RAG/data/documents", glob="*.txt", show_progress=True)
        data = loader.load()
        
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        all_splits = text_splitter.split_documents(data)
        
        embed_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model='text-embedding-3-small'
        )
        
        vector_index = FAISS.from_documents(all_splits, embed_model)
        print("Vector store created successfully")
        
        # 저장 시 디렉토리가 있는지 확인
        os.makedirs("data", exist_ok=True)
        vector_index.save_local("data/vectorstore")
        print("Vector store saved successfully")
        return vector_index
    except Exception as e:
        print(f"Detailed error in create_and_save_vectorstore: {e}")
        return None


# def create_and_save_vectorstore():
#     """챗봇 체인 설정"""
#     try:
#         # 벡터스토어가 이미 있다고 가정하고 로드
#         embed_model = OpenAIEmbeddings(
#             openai_api_key=api_key,
#             model='text-embedding-3-small'
#         )
        
#         vector_index = FAISS.load_local("data/vectorstore", embed_model)
            
#         retriever = vector_index.as_retriever(k=5)
        
    
    # """벡터 스토어 생성 및 저장"""
    # try:
    #     loader = DirectoryLoader("/Users/alookso/Desktop/2025 취준/hw_chatbot_RAG/data/documents", glob="*.txt", show_progress=True)
    #     data = loader.load()
        
    #     from langchain_text_splitters import TokenTextSplitter
    #     text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
    #     all_splits = text_splitter.split_documents(data)
        
    #     embed_model = OpenAIEmbeddings(
    #         openai_api_key=api_key,
    #         model='text-embedding-3-small'
        # )
        
    #     vector_index = FAISS.from_documents(all_splits, embed_model)
    #     vector_index.save_local("data/vectorstore")
    #     return vector_index
    
    
    # except Exception as e:
    #     print(f"Error retrieving vector store: {e}")
    #     return None

def setup_chat_chain():
    """챗봇 체인 설정"""
    try:
        print("Setting up chat chain...")
        # 저장된 벡터 스토어 로드
        embed_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model='text-embedding-3-small'
        )
        
        # 벡터스토어 존재 여부 확인
        if not os.path.exists("data/vectorstore"):
            print("Vector store not found, creating new one...")
            vector_index = create_and_save_vectorstore()
        else:
            print("Loading existing vector store...")
            vector_index = FAISS.load_local(
                "data/vectorstore", 
                embed_model,
                allow_dangerous_deserialization=True
            )
        
        if vector_index is None:
            raise Exception("Failed to initialize vector store")
        
        # # allow_dangerous_deserialization=True 추가
        # vector_index = FAISS.load_local(
        #     "data/vectorstore", 
        #     embed_model,
        #     allow_dangerous_deserialization=True  # 우리가 직접 만든 파일이므로 안전함
        # )
        
        # if os.path.exists("data/vectorstore"):
        #     vector_index = FAISS.load_local("data/vectorstore", embed_model)
        
        # else:
        #     vector_index = create_and_save_vectorstore()
            
        # if not vector_index:
        #     return None
        
        print("Setting up retriever...")
        retriever = vector_index.as_retriever(k=5)
        
        # 프롬프트 템플릿 설정
        template = """
        너는 지금부터 내 페르소나를 가진 챗봇이야. 나는 AI 업계에서 일을 시작하기 위해 일자리를 찾고 있어.
        질문을 하는 사람은 회사의 면접관이고, 항상 예의있으면서도 솔직하고, 정직하게, 자신감있게 대답해줘.
        질문에 답을 하기 어려운 내용이라면 잘 모르겠다고 대답하고, 실제 면접에서 만나게 되면 다시 한번 꼭 물어봐달라고 말해줘.
        대답할 때에는 아래 context를 참고해. context에는 내가 지금까지 했던 경험과 경력을 알려줄께. 
        ###
        {context}
        ###

        {history}

        user: {query}
        답변:  
        """
        

            
        print("Setting up prompt and memory...")   
        prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationBufferWindowMemory(k=3, ai_prefix="job interview")
        
        
        print("Setting up ChatOpenAI...")
        chat = ChatOpenAI(
                    openai_api_key=api_key,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
        
        print("Creating chain...")
        def merge_docs(retrieved_docs):
            return "###\n\n".join([d.page_content for d in retrieved_docs])
        
        chain = RunnableParallel({
                    'context': retriever | merge_docs,
                    'query': RunnablePassthrough(),
                    'history': RunnableLambda(memory.load_memory_variables) | itemgetter('history')
                }) | {
                    'answer': prompt | chat | StrOutputParser(),
                    'context': itemgetter('context'),
                    'prompt': prompt
                }
        print("Chat chain setup completed successfully")
        return chain, memory
                
    except Exception as e:
        print(f"Error in setup_chat_chain: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None













# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# import os
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# # from langchain_community.document_loaders import DirectoryLoader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.chat_models import ChatOpenAI    # 변경됨

# load_dotenv()

# # API 키 확인
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables")

# def create_and_save_vectorstore():
#     """벡터 스토어를 생성하고 저장하는 함수 - 최초 1회만 실행"""
#     print("Creating new vector store...")
#     loader = PyPDFLoader("./data/documents/이력서_김효원.pdf")
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)
#     embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         )
#     vectorstore = FAISS.from_documents(texts, embeddings)
#     # 벡터 스토어 저장
    
#     vectorstore.save_local("vectorstore")
#     print("Vector store saved successfully")
#     return vectorstore


# def setup_rag():
#     """저장된 벡터 스토어를 로드하여 RAG 시스템 구축"""
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         )
        
#         # 저장된 벡터 스토어가 있으면 로드, 없으면 새로 생성
#         if os.path.exists("vectorstore"):
#             vectorstore = FAISS.load_local("vectorstore", embeddings)
#         else:
#             vectorstore = create_and_save_vectorstore()
            
#         if not vectorstore:
#             return None
            
#         llm = ChatOpenAI(temperature=0.7)
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm,
#             retriever=vectorstore.as_retriever(),
#             return_source_documents=True
#         )
        
#         return qa_chain
        
#     except Exception as e:
#         print(f"Error in setup_rag: {e}")
#         return None
    
    