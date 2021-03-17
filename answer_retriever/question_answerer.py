from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import helpers

top_k = 3
path = Path('./data')
model_name = 'retriever_model/distilroberta-base/2021-03-17_21-18-32'
model = SentenceTransformer(model_name)

answers = helpers.read_file(path/'all_answer.txt')

answer_embedding = model.encode(answers, convert_to_tensor=True)

while True:
    print("Please input your question:")
    question = input()
    print('Finding answers ...')
    question_embedding = model.encode([question])
    hits = util.semantic_search(question_embedding, answer_embedding, top_k=top_k)
    print(f'Top {top_k} answers for your question:')
    for i, hit in enumerate(hits[0]):
        print('='*20)
        print(f'{i + 1}st with score {hit["score"]}')
        print(answers[hit['corpus_id']])