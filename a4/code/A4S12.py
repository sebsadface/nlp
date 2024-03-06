# %% [markdown]
# # Assignment 4: Decoding Algorithms (Section 1 and 2)
# 
# - Before running the jupyter notebook, don't forget to copy it into your drive **(`File` => `Save a copy in Drive`)**. *Failing to do this step may result in losing the progress of your code.*
# - **Don't forget to choose "Runtime Type" = GPU in Colab for running this notebook (Runtime > Change Runtime Type > T4 GPU).**
# - For the submission of the assignment, please download this notebook as a **Python file**, named `A4S12.py`.
# 
# - **You will do the following two tasks as detailed under each section, and we will grade both parts:**
#   - **Coding Exercises:** You will complete the the code blocks denoted by **`TODO:`**. We will grade your code in this notebook.
#   - **Questions to Answer:** You will answer questions denoted by **`Q:`** in your write-up.
#   
# ## Section 0: Setup
# 
# Please run all the code blocks in this section. You don't need to implement or change anything here.

# %%
!pip install transformers
!pip install datasets
!pip install evaluate

# %%
"""set device and random seeds"""

######################################################
#  The following helper functions are given to you.
######################################################

from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device: {device}')

def set_seed(seed=19260817):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# %% [markdown]
# ### 0.1 Dataset

# %%
"""load datasets"""

######################################################
#  The following helper code is given to you.
######################################################

from datasets import load_dataset

dataset = load_dataset('Ximing/ROCStories')
train_data, dev_data, test_data = dataset['train'], dataset['validation'], dataset['test']

print(train_data[0])

# %% [markdown]
# ### 0.2 Evaluation Metrics

# %%
"""prepare evaluation"""

######################################################
#  The following helper code is given to you.
######################################################

from evaluate import load
from transformers import RobertaForSequenceClassification, RobertaTokenizer

perplexity_scorer = load("perplexity", module_type="metric")
cola_model_name = "textattack/roberta-base-CoLA"
cola_tokenizer = RobertaTokenizer.from_pretrained(cola_model_name)
cola_model = RobertaForSequenceClassification.from_pretrained(cola_model_name).to(device)

def batchify(data, batch_size):
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

# %%
"""set up evaluation metric"""

######################################################
#  The following helper code is given to you.
######################################################

def compute_perplexity(texts, model='gpt2', batch_size=8):
    score = perplexity_scorer.compute(predictions=texts, add_start_token=True, batch_size=batch_size, model_id=model)
    return score['mean_perplexity']


def compute_fluency(texts, batch_size=8):
  scores = []
  for b_texts in batchify(texts, batch_size):
    inputs = cola_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
      logits = cola_model(**inputs).logits
      probs = logits.softmax(dim=-1)
      scores.extend(probs[:, 1].tolist())
  return sum(scores) / len(scores)


def compute_diversity(texts):
    unigrams, bigrams, trigrams = [], [], []
    total_words = 0
    for gen in texts:
        o = gen.split(' ')
        total_words += len(o)
        for i in range(len(o)):
            unigrams.append(o[i])
        for i in range(len(o) - 1):
            bigrams.append(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.append(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    return len(set(unigrams)) / len(unigrams), len(set(bigrams)) / len(bigrams), len(set(trigrams)) / len(trigrams)


def evaluate(generations, experiment):
  generations = [_ for _ in generations if _ != '']
  perplexity = compute_perplexity(generations)
  fluency = compute_fluency(generations)
  diversity = compute_diversity(generations)
  print(experiment)
  print(f'perplexity = {perplexity:.2f}')
  print(f'fluency = {fluency:.2f}')
  print(f'diversity = {diversity[0]:.2f}, {diversity[1]:.2f}, {diversity[2]:.2f}')
  print()

debug_sents = ["This restaurant is awesome", "My dog is cute and I love it.", "Today is sunny."]
evaluate(debug_sents, 'debugging run')

# %% [markdown]
# ### 0.3: Load Model

# %%
"""load model and tokenizer"""

######################################################
#  The following helper code is given to you.
######################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# %% [markdown]
# ## **Section 1: Basic Decoding Algorithms**
# 
# In this section, you will implement a few basic decoding algorithms:
# 1. Greedy decoding
# 2. Vanilla sampling
# 3. Temperature sampling
# 4. Top-k sampling
# 5. Top-p sampling
# 
# We have provided a wrapper function `decode()` that takes care of batching, controlling max length, and handling the EOS token.
# You will be asked to implement the core function of each method: *given the pre-softmax logits of the next token, decide what the next token is.*
# 
# **The wrapper calls the core function of each decoding algorithm, which you will implement in the subsections below.**

# %%
"""decode main wrapper function"""

######################################################
#  The following helper code is given to you.
######################################################

def decode(prompts, max_len, method, **kwargs):
  encodings_dict = tokenizer(prompts, return_tensors="pt", padding=True)
  input_ids = encodings_dict['input_ids'].to(device)
  attention_mask = encodings_dict['attention_mask'].to(device)

  model_kwargs = {'attention_mask': attention_mask}
  batch_size, input_seq_len = input_ids.shape

  unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

  for step in range(max_len):
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    with torch.no_grad():
      outputs = model(**model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False)

    if step == 0:
      last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
      next_token_logits = outputs.logits[range(batch_size), last_non_masked_idx, :]
    else:
      next_token_logits = outputs.logits[:, -1, :]

    log_prob = F.log_softmax(next_token_logits, dim=-1)

    if method == 'greedy':
      next_tokens = greedy(next_token_logits)
    elif method == 'sample':
      next_tokens = sample(next_token_logits)
    elif method == 'temperature':
      next_tokens = temperature(next_token_logits, t=kwargs.get('t', 0.8))
    elif method == 'topk':
      next_tokens = topk(next_token_logits, k=kwargs.get('k', 20))
    elif method == 'topp':
      next_tokens = topp(next_token_logits, p=kwargs.get('p', 0.7))

    # finished sentences should have their next token be a padding token
    next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)

    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)

    # if eos_token was found in one sentence, set sentence to finished
    unfinished_sequences = unfinished_sequences.mul((next_tokens != tokenizer.eos_token_id).long())

    if unfinished_sequences.max() == 0:
      break

  response_ids = input_ids[:, input_seq_len:]
  response_text = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in response_ids]

  return response_text

# %%
"""debug helper code"""

######################################################
#  The following helper code is given to you.
######################################################

# For debugging, we duplicate a single prompt 10 times so that we obtain 10 generations for the same prompt
dev_prompts = [dev_data[0]['prompt']] * 10

def print_generations(prompts, generations):
  for prompt, generation in zip(prompts, generations):
    print(f'{[prompt]} ==> {[generation]}')

# %% [markdown]
# ### 1.1: Greedy Decoding

# %%
def greedy(next_token_logits):
  '''
  inputs:
  - next_token_logits: Tensor(size = (B, V), dtype = float)
  outputs:
  - next_tokens: Tensor(size = (B), dtype = long)
  '''

  # TODO: compute `next_tokens` from `next_token_logits`.
  # Hint: use torch.argmax()
  next_tokens = torch.argmax(next_token_logits, dim=-1)

  return next_tokens


generations = decode(dev_prompts, max_len=20, method='greedy')
print_generations(dev_prompts, generations)

# %% [markdown]
# ### 1.2: Vanilla Sampling and Temperature Sampling

# %%
def sample(next_token_logits):
  '''
  inputs:
  - next_token_logits: Tensor(size = (B, V), dtype = float)
  outputs:
  - next_tokens: Tensor(size = (B), dtype = long)
  '''

  # TODO: compute the probabilities `probs` from the logits.
  # Hint: `probs` should have size (B, V)
  probs = F.softmax(next_token_logits, dim=-1)

  # TODO: compute `next_tokens` from `probs`.
  # Hint: use torch.multinomial()
  next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

  return next_tokens


set_seed()
generations = decode(dev_prompts, max_len=20, method='sample')
print_generations(dev_prompts, generations)

# %%
def temperature(next_token_logits, t):
  '''
  inputs:
  - next_token_logits: Tensor(size = (B, V), dtype = float)
  - t: float
  outputs:
  - next_tokens: Tensor(size = (B), dtype = long)
  '''

  # TODO: compute the probabilities `probs` from the logits, with temperature applied.
  probs = F.softmax(next_token_logits / t, dim=-1)

  # TODO: compute `next_tokens` from `probs`.
  next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

  return next_tokens


set_seed()
generations = decode(dev_prompts, max_len=20, method='temperature', t=0.8)
print_generations(dev_prompts, generations)

# %% [markdown]
# ### 1.3: Top-k Sampling

# %%
def topk(next_token_logits, k):
  '''
  inputs:
  - next_token_logits: Tensor(size = (B, V), dtype = float)
  - k: int
  outputs:
  - next_tokens: Tensor(size = (B), dtype = long)
  '''

  # TODO: Keep only top-k tokens with highest probabilities.
  # Hint: use torch.topk()
  topk_logits, topk_indices = torch.topk(next_token_logits, k, dim=-1)

  # Create a mask to zero out all logits not in top-k
  indices_to_remove = next_token_logits < topk_logits[:, -1].unsqueeze(1)

  # Mask the logits
  next_token_logits[indices_to_remove] = float('-inf')

  # TODO: Sample from the masked logits
  probs = F.softmax(next_token_logits, dim=-1)
  next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

  return next_tokens


set_seed()
generations = decode(dev_prompts, max_len=20, method='topk', k=20)
print_generations(dev_prompts, generations)

# %% [markdown]
# ### 1.4: Top-p Sampling

# %%
def topp(next_token_logits, p):
  '''
  inputs:
  - next_token_logits: Tensor(size = (B, V), dtype = float)
  - p: float
  outputs:
  - next_tokens: Tensor(size = (B), dtype = long)
  '''

  # TODO: Sort the logits in descending order, and compute
  # the cumulative probabilities `cum_probs` on the sorted logits
  sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
  sorted_probs = F.softmax(sorted_logits, dim=-1)
  cum_probs = torch.cumsum(sorted_probs, dim=-1)

  # Create a mask to zero out all logits not in top-p
  sorted_indices_to_remove = cum_probs > p
  sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
  sorted_indices_to_remove[:, 0] = 0
  # Restore mask to original indices
  indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

  # Mask the logits
  next_token_logits[indices_to_remove] = float('-inf')

  # TODO: Sample from the masked logits
  probs = F.softmax(next_token_logits, dim=-1)
  next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

  return next_tokens


set_seed()
generations = decode(dev_prompts, max_len=20, method='topp', p=0.7)
print_generations(dev_prompts, generations)

# %% [markdown]
# ### 1.5: Evaluate!
# 
# Run the following cell to obtain the evaluation results, which you should include in your writeup.
# Also don't forget to answer the questions.

# %%
prompts = [item['prompt'] for item in test_data][:10]
GENERATIONS_PER_PROMPT = 10
MAX_LEN = 100

for experiment in ['greedy', 'sample', 'temperature', 'topk', 'topp']:
  generations = []
  for prompt in tqdm(prompts):
    generations += decode([prompt] * GENERATIONS_PER_PROMPT, max_len=MAX_LEN, method=experiment)
  evaluate(generations, experiment)

# %% [markdown]
# ## **Section 2: Beam Search**
# 
# - In this part of the assignment, your main task is to implement beam search algorithm. While debugging for implementation, we recommend setting `device` to `cpu`, as debugging would not require too much resource for this assignment. Once you are done with implementation, you may switch `device` to `cuda` and choose `Colab Runtime Type = GPU` for faster evaluation of your algorithm.
# 
# - In the following part, we have provided some helpful functions and classes. Please make sure that you understand them. But you don't need to implement/change anything of them until **Section 2.1**.

# %% [markdown]
# ### Configurations: load model and tokenizer

# %%
######################################################
#  The following helper code is given to you.
######################################################

from dataclasses import dataclass
from typing import List, Tuple

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
tokenizer.padding_side = "left"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

# %% [markdown]
# ### Helper classes: `BeamHypothesisList` and `BeamManager`

# %%
######################################################
#  The following helper code is given to you.
#  Try your best to understand the code,
#  but you don't need to implement anything here.
######################################################

@dataclass
class BeamHypothesis:
    def __init__(self, input_ids: torch.LongTensor, score: torch.FloatTensor | float):
        self.input_ids: torch.LongTensor = input_ids  # a single token sequence of size (seq_len,)
        self.score: torch.FloatTensor = score  # a scalar score for the token sequence

    def __str__(self):
        return f"BeamHypothesis(input_ids: {self.input_ids}, score: {self.score})"

class BeamHypothesisList:
    def __init__(self, num_beams: int):
        self.beam_hypotheses: List[BeamHypothesis] = []  # list of beam_hypothesis
        self.num_beams: int = num_beams

        self.worst_score = 1e9  # worst beam score in self.beam_hypotheses

    def add(self, new_input_ids: torch.LongTensor, sum_logprobs: float):
        """
        :param new_input_ids: new token sequence of size (1, seq_len)
        :param sum_logprobs: sum of log probabilities of tokens in new_input_ids
        Given a new hypothesis (new_input_ids) and its corresponding sum_logprobs, update self.beam_hypotheses with a finished hypothesis.
        (1) If the new_input_ids has higher score than the current worst score in self.beam_hypotheses,
            we replace the worst one with the new hypothesis.
        (2) Otherwise, we do not make any change to self.beam_hypotheses.
        """
        # For score, we compute average token log probability
        score = sum_logprobs / new_input_ids.size(-1)

        # add the new hypothesis if we still have vacant beams
        if len(self.beam_hypotheses) < self.num_beams:
            # Initialize the new_beam_hypothesis using new_input_ids and score
            new_beam_hypothesis: BeamHypothesis = BeamHypothesis(input_ids=new_input_ids, score=score)

            # Add new_beam_hypothesis to beam_hypotheses
            self.beam_hypotheses.append(new_beam_hypothesis)

        # even if the beam_hypotheses are full, if the new hypothesis has higher score than the worst hypothesis, we replace the worst hypothesis
        elif score > self.worst_score:
            # Remove the worst hypothesis, the one with the lowest score in self.beam_hypotheses
            worst_hypothesis: BeamHypothesis = min(self.beam_hypotheses, key=lambda hyp: hyp.score)
            self.beam_hypotheses.remove(worst_hypothesis)

            # Add new hypothesis
            # Initialize the new_beam_hypothesis using new_input_ids and score
            new_beam_hypothesis = BeamHypothesis(input_ids=new_input_ids, score=score)

            # Add new_beam_hypothesis to beam_hypotheses
            self.beam_hypotheses.append(new_beam_hypothesis)

        # Update the worst score - the lowest score among all beams in self.beam_hypotheses
        self.worst_score = min(float(hyp.score) for hyp in self.beam_hypotheses)

        # Sanity check
        assert len(self.beam_hypotheses) <= self.num_beams

# %%
######################################################
#  The following helper code is given to you.
#  Try your best to understand the code,
#  but you don't need to implement anything here.
######################################################

class BeamManager:
    def __init__(self, batch_size: int, num_beams: int):
        self.finished_beam_hypotheses_list = [BeamHypothesisList(num_beams) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.num_beams = num_beams

    def process(self,
                input_ids: torch.LongTensor,
                top_token_scores: torch.FloatTensor,
                top_token_indices: torch.LongTensor,
                top_token_beam_indices: torch.LongTensor
                ):
        """
        :param input_ids: (batch_size * num_beams, current_seq_length), the input_ids that were used to compute top_tokens
        :param top_token_scores: (batch_size, 2 * num_beams), representing the score of each top token
        :param top_token_indices: (batch_size, 2 * num_beams), representing each token's index (in vocabulary) of the top tokens
        :param top_token_beam_indices: (batch_size, 2 * num_beams), representing each token's corresponding beam index of the top tokens

        Note: the input arguments `top_token_*` for each sample in batch are sorted from the largest score to the smallest score.
        For example, if batch_size = 2 and num_beams = 3, then each of these values denote
        top_token_indices[1, 2]: what is the third-best next token for the second sample in the batch?
        top_token_scores[0, 1]: what is the score of the second-best next token for the first sample in the batch?
        top_token_beam_indices[0, 1]: which beam did we use to generate the second-best next token for the first sample in the batch?

        In this function, for each of the top-(2 * num_beams) tokens, we do the following:
        (1) If the top token is EOS token:
            This means that this hypothesis is done. Therefore, we save the hypothesis so-far to self.finished_beam_hypotheses_list.
        (2) If the top token is not EOS token
            We have to keep searching with this hypothesis. Therefore, we prepare the hypothesis for next time step.

        Returns a dictionary, where
        "unfinished_scores": size (batch_size * num_beams,), the score of the unfinished beams
        "unfinished_token_indices": size (batch_size * num_beams,), the index of the last token in the unfinished beams
        "unfinished_beam_indices": the index of the beam that was used to generate the new unfinished beam
        """
        device = top_token_scores.device

        # Initialize unfinished_token_*, which we will return for the next time step.
        unfinished_scores = torch.zeros((self.batch_size, self.num_beams), dtype=top_token_scores.dtype).to(device)  # score of the unfinished beams
        unfinished_token_indices = torch.zeros((self.batch_size, self.num_beams), dtype=top_token_indices.dtype).to(
            device)  # index of the last token of the unfinished beams
        unfinished_token_beam_indices = torch.zeros((self.batch_size, self.num_beams), dtype=top_token_beam_indices.dtype).to(
            device)  # index of the unfinished beam in the batch

        # Loop over the batch
        for batch_idx in range(self.batch_size):
            # Get sample_beam_hypothesis_list: the finished_beam_hypothesis_list for this sample in the batch
            sample_beam_hypothesis_list: BeamHypothesisList = self.finished_beam_hypotheses_list[batch_idx]

            # Get the top_token_scores, top_token_indices, top_token_beam_indices for this sample in the batch
            # NOTE: size of sample_top_token_*: (2 * num_beams,)
            sample_top_token_scores = top_token_scores[batch_idx]
            sample_top_token_indices = top_token_indices[batch_idx]
            sample_top_token_beam_indices = top_token_beam_indices[batch_idx]

            # Loop over all top tokens
            sample_beam_idx = 0
            for top_token_score, top_token_index, top_token_beam_index in zip(
                    sample_top_token_scores, sample_top_token_indices, sample_top_token_beam_indices
            ):
                # Note that top_token_beam_indices only denotes the index of the beam in each sample.
                # We transform this into `beam_idx_in_batch`, where we denote the index of the beam among all (batch_size * num_beams) beams in the batch.
                beam_idx_in_batch = batch_idx * self.num_beams + top_token_beam_index

                # if top_token == EOS, we add the generation so-far to the beam_hypotheses_list
                if top_token_index.item() == eos_token_id:
                    # Among the (batch_size * num_beams) input_ids, find the input_ids that correspond to this top_token
                    # NOTE: the size of new_input_ids: (seq_len,)
                    new_input_ids = input_ids[beam_idx_in_batch]

                    # Add the new beam to sample_beam_hypothesis_list
                    sample_beam_hypothesis_list.add(
                        new_input_ids,
                        top_token_score
                    )

                # if top_token =/= EOS, we aggregate them for next time step.
                else:
                    # Store the score, token_index, beam_idx_in_batch to the unfinished_scores, unfinished_token_indices, unfinished_token_beam_indices
                    unfinished_scores[batch_idx, sample_beam_idx] = top_token_score
                    unfinished_token_indices[batch_idx, sample_beam_idx] = top_token_index
                    unfinished_token_beam_indices[batch_idx, sample_beam_idx] = beam_idx_in_batch

                    sample_beam_idx += 1

                # once we have `num_beams` number of new beams, we don't have to add anymore.
                if sample_beam_idx == self.num_beams:
                    break

        # Return the dictionary of unfinished_scores, unfinished_token_indices, unfinished_beam_indices
        # Make sure to change the size of each tensor to (batch_size * num_beams,)
        return {
            "unfinished_scores": unfinished_scores.view(-1),  # (batch_size * num_beams,)
            "unfinished_token_indices": unfinished_token_indices.view(-1),  # (batch_size * num_beams,)
            "unfinished_beam_indices": unfinished_token_beam_indices.view(-1),  # (batch_size * num_beams,)
        }

    def finalize(
            self,
            input_ids: torch.LongTensor,
            beam_scores: torch.FloatTensor,
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
        """
        :param input_ids: (batch_idx * num_beams, max_length), input_ids of unfinished beams
        :param beam_scores: (batch_idx * num_beams,), scores of unfinished beams
        Get the final best beams, among
        (1) unfinished beams, for which we get the input_ids and beam_scores as arguments
        (2) finished beams, which we store in self.batch_beam_hypothesis_list
        Returns a tuple of two lists, where
        - tuple[0] is the list of the input_ids of the best beams (length: batch_idx)
        - tuple[1] is the list of the scores of the best beams (length: batch_idx
        """

        # 1. Add all unfinished beam hypotheses to self.finished_beam_hypotheses_list
        for batch_idx in range(self.batch_size):
            # Get sample_beam_hypothesis_list: the finished_beam_hypothesis_list for this sample in the batch
            sample_beam_hypothesis_list: BeamHypothesisList = self.finished_beam_hypotheses_list[batch_idx]

            for sample_beam_idx in range(self.num_beams):
                # Get beam_idx_in_batch: index of the beam in all `batch_size * num_beams` beams in the batch
                beam_idx_in_batch = batch_idx * self.num_beams + sample_beam_idx

                # Get the input_id for this beam, using `beam_idx_in_batch`
                # NOTE: the size of new_input_ids: (seq_len,)
                new_input_ids = input_ids[beam_idx_in_batch]

                # Get the score of this beam, using `beam_idx_in_batch`
                # NOTE: beam_score should be a scalar
                beam_score = beam_scores[beam_idx_in_batch].item()

                # Add the new hypothesis to sample_beam_hypothesis_list
                sample_beam_hypothesis_list.add(new_input_ids, beam_score)

        # 2. Select the best hypothesis from each beam_hypothesis_list
        best_input_ids = []
        best_scores = []
        for batch_idx in range(self.batch_size):
            # Get sample_beam_hypothesis_list: the finished_beam_hypothesis_list for this sample in the batch
            sample_beam_hypothesis_list: BeamHypothesisList = self.finished_beam_hypotheses_list[batch_idx]

            # Get best_hypothesis among sample_beam_hypothesis_list (the one with the highest score)
            best_hypothesis = max(sample_beam_hypothesis_list.beam_hypotheses, key=lambda hyp: hyp.score)

            # Save the input_ids and score of best_hypothesis
            best_input_ids.append(best_hypothesis.input_ids)
            best_scores.append(best_hypothesis.score)

        return best_input_ids, best_scores

# %% [markdown]
# ### 2.1 Beam Search
# 
# **TODO: fill in the todos in `beam_search`.**

# %%
def beam_search(prompts: List[str], num_beams: int, max_length: int) -> List[str]:
    """
    :param prompts: list of prompt strings
    :param num_beams: number of beams
    :param max_length: max generation length
    :return: list of generation, including both the original prompt and generation
    """
    # TODO: encode the prompts using tokenizer, to get input_ids and attention_mask
    input_encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids, attention_mask = input_encoding["input_ids"].to(device), input_encoding["attention_mask"].to(device)

    if input_ids.size(-1) > max_length:
      raise ValueError("Input ID is larger than max_length.")

    # --- Do not change below --- #
    batch_size = input_ids.size(0)
    vocab_size = len(tokenizer)

    # initialize model_kwargs
    model_kwargs = {'attention_mask': attention_mask}

    # interleave input_ids according to num_beams.
    # For example, input_ids for ["Hi", "good"] with num_beams=3 becomes ["Hi", "Hi", "Hi", "good", "good", "good"]
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=num_beams,
        is_encoder_decoder=False,
        **model_kwargs,
    )
    # NOTE: the type of `input_ids` and `model_kwargs` are as following:
    # input_ids: tensor of size (batch_size * num_beams, seq_len)
    # model_kwargs: a dictionary with single element 'attention_mask', sized (batch_size * num_beams, seq_len)
    # --- Do not change above --- #

    # TODO: initialize beam_manager
    beam_manager = BeamManager(batch_size, num_beams)

    # TODO: initialize unfinished_beam_scores, a tensor of size (batch_size, num_beams) with all elements = 0
    unfinished_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float).to(device)

    # For each sample in the batch, set all initial beam_score to -1e9, except for the first beam
    unfinished_beam_scores[:, 1:] = -1e9
    unfinished_beam_scores = unfinished_beam_scores.view(-1)  # (batch_size * num_beams,)

    while True:
        # --- Do not change below --- #
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # --- Do not change above --- #

        # TODO: run model forward pass with model_inputs as the input
        # NOTE: we should set return_dict=True, output_attentions=False and output_hidden_states=False
        model_outputs = model(**model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False)

        # TODO compute log_probs for next tokens, using `model_outputs.logits`
        # NOTE: size of next_token_scores: (batch_size * num_beams, vocab_size)
        next_token_logits = model_outputs.logits[:, -1, :]
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        # TODO add previous beam_scores to the next_token_scores
        # NOTE: size of new_scores: # (batch_size * num_beams, vocab_size)
        new_scores = next_token_scores + unfinished_beam_scores.unsqueeze(-1)

        # TODO: retrieve top-(2 * num_beams) next tokens for each sample in the batch
        # NOTE: size of `top_token_scores` and `top_token_indices` needs to be: (batch_size, 2 * num_beams)
        # NOTE: `top_token_scores` and `top_token_indices` should be sorted from the one with larget score to the one with smallest score (for each sample in batch)
        # Hint: use torch.topk with largest=True, sorted=True
        # Hint: new_scores needs to be transformed to shape (batch_size, num_beams * vocab_size) prior to topk operation.
        top_token_scores, top_token_indices = torch.topk(new_scores.view(batch_size, -1), 2 * num_beams, dim=-1, largest=True, sorted=True)

        # Since top_token_indices are over num_beams * vocab_size, divide it by num_beams to get vocabulary index and beam index
        top_token_beam_indices = torch.div(top_token_indices, vocab_size,
                                           rounding_mode="floor")  # from which beam the top-token was retrieved from
        top_token_indices = top_token_indices % vocab_size  # the index of top-token in the vocabulary

        # TODO: Run beam_manager.process and save the results in unfinished_beam_scores, unfinished_token_indices and unfinished_beam_indices
        unfinished_beam_outputs = beam_manager.process(input_ids, top_token_scores, top_token_indices, top_token_beam_indices)
        unfinished_beam_scores = unfinished_beam_outputs["unfinished_scores"]
        unfinished_token_indices = unfinished_beam_outputs["unfinished_token_indices"]
        unfinished_beam_indices = unfinished_beam_outputs["unfinished_beam_indices"]

        # --- Prepare input_ids for next time step --- #
        # TODO: index input_ids with the unfinished beam indices
        # NOTE: input_ids should be (batch_size * num_beams, seq_len)
        input_ids = input_ids[unfinished_beam_indices]

        # TODO: concatenate the unfinished_token_indices to the input_ids
        input_ids = torch.cat([input_ids, unfinished_token_indices.unsqueeze(-1)], dim=-1)

        # --- Do not change below --- #
        # update the model_kwargs according to the concatenated input_ids
        model_kwargs = model._update_model_kwargs_for_generation(
            model_outputs, model_kwargs, is_encoder_decoder=False
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = model._temporary_reorder_cache(
                model_kwargs["past_key_values"], unfinished_beam_indices,
            )
        # --- Do not change above --- #

        # if unfinished input_ids reach the max seq length, exit the loop
        if input_ids.size(-1) == max_length:
            break

    # TODO: Run beam_manager.finalize to get the best_input_ids and best_scores, among all finished / unfinished beams
    best_input_ids, best_scores = beam_manager.finalize(input_ids, unfinished_beam_scores)

    # if len(best_input_ids) < max_length, pad them to the max length
    for batch_idx, sample_input_ids in enumerate(best_input_ids):
        if sample_input_ids.size(-1) < max_length:
            pad_tensor = torch.LongTensor([pad_token_id] * (max_length - sample_input_ids.size(-1))).to(device)

            # TODO: pad best_input_ids with pad_tensor
            best_input_ids[batch_idx] = torch.cat([sample_input_ids, pad_tensor], dim=-1)

    # TODO: transform best_input_ids (which is currently a list of tensors) into a tensor of size (batch_idx, max_seq_length)
    # Hint: use torch.stack
    best_input_ids = torch.stack(best_input_ids, dim=0)

    return tokenizer.batch_decode(best_input_ids, skip_special_tokens=True)

# %% [markdown]
# ### Sanity check for debugging

# %%
sents = [
    "This restaurant is awesome.",
    "My dog is cute and I love it.",
    "Today is sunny.",
]

beam_search(sents, num_beams=5, max_length=15)

# expected output: ["This restaurant is awesome. I've been here a few", 'My dog is cute and I love it.\n\nRated 5 out of', 'Today is sunny. The sun is shining. The']

# %% [markdown]
# ### 2.2 Evaluate Beam Search!

# %%
# If your implementation is efficient enough, the following code will run in no longer than 3 minutes with device = gpu.

from pprint import pprint

prompts = [item['prompt'] for item in test_data][:100]
MAX_LEN = 100
NUM_BEAMS = 5
BATCH_SIZE = 5

generations = []
for batch_start_idx in tqdm(range(0, len(prompts), BATCH_SIZE)):
  batched_prompts = prompts[batch_start_idx: batch_start_idx + BATCH_SIZE]
  batched_generations = beam_search(batched_prompts, NUM_BEAMS, MAX_LEN)

  # remove prompt from generation
  batched_generations = [generation[len(prompt):] for prompt, generation in zip(batched_prompts, batched_generations)]

  generations += batched_generations

evaluate(generations, 'Beam Search')

# %%
# print first 10 generations

sampled_prompts = prompts[:10]
sampled_generations = generations[:10]

for idx, (prompt, generation) in enumerate(zip(sampled_prompts, sampled_generations)):
  print(f"Prompt {idx}")
  print(prompt)
  print(f"Generation {idx}")
  print(generation)
  print("---------")
  print()


