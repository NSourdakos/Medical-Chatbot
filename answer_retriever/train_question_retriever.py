from sentence_transformers import SentenceTransformer, util, models, evaluation, losses, InputExample
from pathlib import Path
from datetime import datetime
import helpers
from torch.utils.data import DataLoader

model_name = 'retriever_model/distilroberta-base/2021-03-17_21-18-32'

train_batch_size = 32

base_model_name = 'distilroberta-base'

def create_model(base_model_name):
    word_embedding_model = models.Transformer(base_model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model = SentenceTransformer(model_name)

#Train a new one
# model = create_model(base_model_name)

output_path = f'./retriever_model/{model_name.replace("/", "-")}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

data_path = Path('./data')

train_dataset = helpers.load_training_dataset(data_path)
val_dataset = helpers.load_val_dataset(data_path)

acc_evaluator = helpers.RetrievalEvaluator(val_dataset[0], val_dataset[1], top_k=1)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=acc_evaluator,
        epochs=5,
        warmup_steps=1000,
        output_path=output_path,
        evaluation_steps=1000,
        use_amp=True
)

model.save(output_path)
