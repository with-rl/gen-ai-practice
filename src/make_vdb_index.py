import json
import argparse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dump_file", type=str, default="data/chunk_db.json")
    p.add_argument("--vdb_index", type=str, default="data/faiss_kowiki_vdb")
    p.add_argument(
        "--context_embedding_id",
        type=str,
        default="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    )

    args = p.parse_args()

    return args


def main(args):
    # full chunks 읽어오기
    full_chunks = []
    with open(args.dump_file) as f:
        for line in f:
            row = json.loads(line)
            full_chunks.append(row['chunk'])

    # make faiss index
    embed_model = HuggingFaceEmbeddings(
        model_name=args.context_embedding_id,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_texts(full_chunks, embed_model)

    # save faiss index
    db.save_local(args.vdb_index)


if __name__ == "__main__":
    args = get_args()
    main(args)

