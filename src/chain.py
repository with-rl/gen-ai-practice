import os
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from langserve import CustomUserType


PROMPT_TEMPLATE = """당신이 가진 지식보다 아래 내용을 내용을 참고해서 '질문'에 대해서 답변해 주세요.:

{context}

질문: {question}
"""


class Question(CustomUserType):
    question: str


class FAISSChain:
    def __init__(self, args):
        # loading vdb index
        embed_model = HuggingFaceEmbeddings(
            model_name=args.question_embedding_id,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        db = FAISS.load_local(
            args.vdb_index, embed_model, allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": args.top_k})

        # llm
        llm = HuggingFaceEndpoint(
            repo_id=args.llm_model_id,
            max_new_tokens=1024,
            temperature=0.1,
        )

        # prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        self.chain = (
            RunnableLambda(self.custom_input)
            | {
                "context": retriever | self.format_contexts,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

    def custom_input(self, question: Question) -> str:
        assert isinstance(question, Question)
        return question.question

    # contxt formatt
    def format_contexts(self, contexts):
        return "\n\n".join(
            [
                d.page_content.replace("&lt;", "<").replace("&gt;", ">").strip()
                for d in reversed(contexts)
            ]
        )

