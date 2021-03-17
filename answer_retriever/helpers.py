from pathlib import Path
from sentence_transformers import InputExample, util, evaluation
 
def read_file(path):
    lines = [line.rstrip() for line in open(path, 'rt')]
    return lines

def read_qa_pairs(path, dataset):
    assert dataset in ['train', 'test', 'validate']
    questions = read_file(path/f'{dataset}_question.txt')
    answers = read_file(path/f'{dataset}_question.txt')
    return questions, answers


def load_training_dataset_asym(path):
    questions, answers = read_qa_pairs(path, dataset)
    dataset = [InputExample(texts={'question': question, 'answer': answer}, label=1) for question, answer in zip(questions, answers)]
    return dataset


def load_training_dataset(path):
    questions, answers = read_qa_pairs(path, 'train')
    dataset = [InputExample(texts=[question, answer], label=1.0) for question, answer in zip(questions, answers)]

    return dataset

def load_val_dataset_asym(path):
    questions = read_file(path/'validate_question.txt')
    answers = read_file(path/'validate_answer.txt')
    questions = [{'question': question} for question in questions]
    answers = [{'answer': answer} for answer in answers]
    scores = [1.] * len(questions)
    return questions, answers, scores


def load_val_dataset(path):
    questions, answers = read_qa_pairs(path, 'validate')
    scores = [1.] * len(questions)
    return questions, answers, scores


def cal_matching_accuracy(hits):
    correct_match = 0
    for q_id, matches in enumerate(hits):
        corpus_ids = [match['corpus_id'] for match in matches]
        if q_id in corpus_ids:
            correct_match += 1
    return correct_match



def evaluate(model, questions, answers, top_k):
    question_embedding = model.encode(questions, convert_to_tensor=True)
    answer_embedding = model.encode(answers, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, answer_embedding, top_k=top_k)
    correct_match = cal_matching_accuracy(hits)
    accuracy = correct_match / len(questions)
    return accuracy




class RetrievalEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, questions, answers, top_k):
        self.questions = questions
        self.answers = answers
        self.top_k = top_k
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        print('Start evaluation')
        accuracy = evaluate(model, self.questions, self.answers, self.top_k)
        print(f"Epoch {epoch} Step: {steps} accuracy: {accuracy}")
        return accuracy