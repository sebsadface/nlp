# %% [markdown]
# # Assignment 3: LM Alignment
# 
# - Before running the jupyter notebook, don't forget to copy it into your drive **(`File` => `Save a copy in Drive`)**. *Failing to do this step may result in losing the progress of your code.*
# - **Don't forget to choose "Runtime Type" = GPU in Colab for running this notebook (Runtime > Change Runtime Type > T4 GPU).**
# - For the submission of the assignment, please download this notebook as a **Python file**, named `A3.py`.
# 
# - There are four sections for this exercise, mapping to the four sections in your A3 instruction file. **You will do the following two tasks as detailed under each section, and we will grade both parts:**
#   - **Coding Exercises:** You will complete the the code blocks denoted by **`TODO:`**. We will grade your code in this notebook.
#   - **Questions to Answer:** You will answer questions denoted by **`Q:`** in your write-up.

# %% [markdown]
# # Section 1: Setup and Baseline Evaluation
# 

# %% [markdown]
# ### 1.0 Preparation

# %%
! pip install datasets
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AdamW, set_seed
from typing import Dict, Union, List, Tuple

# %% [markdown]
# ## **Coding Exercises** for Section 1:
# **You will complete the following code blocks denoted by `TODO:`.**

# %% [markdown]
# ### 1.1 Loading the base model, the reward model, and data

# %%
"""
Load initial model and tokenizer and send it to GPU if available.

Since you will be batching and GPT2 is a decoder-only architecture,
load your tokenizer with padding_side='left' to ensure
that all the <pad> tokens appear before the tokenized text.

To avoid warnings, consider setting pad_token to its eos_token.
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', device)
model_name = 'distilgpt2'

# TODO: Load the tokenizer.
# Hint: remember to add paddings on the left for the decoder-only model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# TODO: set the pad_token to the eos_token.
tokenizer.pad_token = tokenizer.eos_token
# TODO: Load the model and sent it to the device.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.config.pad_token_id = tokenizer.eos_token_id

# %%
"""
Load the reward model.
"""

# TODO: Load reward model and tokenizer from huggingface (use "omidroshani/imdb-sentiment-analysis").
# Hint: Remember to send the model to the device.
reward_tokenizer = AutoTokenizer.from_pretrained("omidroshani/imdb-sentiment-analysis")
reward_model = AutoModelForSequenceClassification.from_pretrained("omidroshani/imdb-sentiment-analysis").to(device)

# %%
"""
Load data.
"""

# TODO: Load IMDB dataset (https://huggingface.co/datasets/imdb) train & test splits.
dataset = load_dataset('imdb')
train_texts = dataset['train']['text']
test_texts = dataset['test']['text']

random.seed(42)

# TODO: shuffle the samples within every data split.
random.shuffle(train_texts)
random.shuffle(test_texts)


# %% [markdown]
# ### 1.2 Evaluation.

# %%
def compute_reward_for_mini_batch(batch,
                                  tokenizer, model,
                                  reward_tokenizer, reward_model,
                                  seed_token_length=10, max_new_tokens=20):
  """
  Write an evaluation function that for each review in the test set it
    1. tokenizes the review.
    2. truncates the review to 10 tokens (seed_token_length).
    3. Given the truncated seed review, randomly generate 20 more tokens with
       the LM so that the final review has a length of 30 tokens.
    4. Get the probability of the generated review being positive.

  The evaluation function should return the final average rewards and the text generations.
  """

  # TODO: tokenize the review, truncating it to 10 tokens (i.e., seed_token_length).
  # Hint: remember to send the encoded text to the device.
  input_tokens = tokenizer(batch, truncation=True, max_length=seed_token_length, return_tensors="pt", padding='max_length').to(device)


  # TODO: randomly generate 20 more tokens (i.e., max_new_tokens) with the LM so
  # that the final review has a length of 30 tokens.
  output_tokens = model.generate(input_tokens['input_ids'].to(device), attention_mask=input_tokens['attention_mask'].to(device), max_length=seed_token_length + max_new_tokens, pad_token_id=tokenizer.eos_token_id)



  # TODO: decode the generated texts and store them in a list.
  generated = [tokenizer.decode(output, skip_special_tokens=True).replace('\n', ' ').strip() for output in output_tokens]

  # TODO: get the probability of the generated review being positive.
  # Hint: you will need to encode the generated text, feed it through the reward model,
  # and get the probablity of the output scores.
  encoded = reward_tokenizer(generated, return_tensors="pt", padding=True, truncation=True)
  output = reward_model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))
  probs = F.softmax(output.logits, dim=-1)
  reward = probs[:, 1]

  return reward, output_tokens, generated


def evaluate(tokenizer, model, reward_tokenizer, reward_model, batch_size=32):
  """
  Code an evaluation loop that simultaneously evaluates a mini-batch
  of size batch_size. Use the helper function compute_reward_for_mini_batch().
  """
  set_seed(42)
  with torch.no_grad():
    test_batches = len(test_texts) // batch_size

    # Evaluate the whole dataset by looping through
    # mini-batch reward computations and accumulate total_reward.
    total_reward = 0
    generations = []
    loop = tqdm(total=test_batches, position=0, leave=False)
    for i in range(0, len(test_texts), batch_size):
        # TODO: get the current batch of data.
        batch = test_texts[i:i+batch_size]

        # TODO: use compute_reward_for_mini_batch() for evaluating each mini batch.
        reward, _, generated = compute_reward_for_mini_batch(batch, tokenizer, model, reward_tokenizer, reward_model)

        # TODO: add generated to generations.
        generations.extend(generated)
        # TODO: add average batch reward to total_reward.
        total_reward += reward.mean().item() * len(batch)
        # TODO: compute the average_reward so far for the display of the progress bar.
        average_reward = total_reward / (i + batch_size)

        loop.set_description(f"Average Reward: {average_reward:.4f}")
        loop.update(1)

    # TODO: compute the final_average_reward.
    final_average_reward = total_reward / len(test_texts)

    print(f"Final Average Reward: {final_average_reward:.4f}")
    print(f"Example texts: {generations[:5]}")

    return {
        'reward': final_average_reward,
        'generations': generations,
    }

results = evaluate(tokenizer, model, reward_tokenizer, reward_model)
print("Reward:", results["reward"])

# %% [markdown]
# ## **Questions to Answer** for Section 1:
# **Answer these questions in your write-up report.**
# - **Q1.1:** Run the pretrained model on the evaluation funciton over the entire test set. What is the average reward?
#   - *Hint:* On a free Google Colab T4 instance, with a generation batch size of 32, this should take about 5 minutes to run.
# - **Q1.2:** How do you feel about the quality of the generations? What sentiment do they express? Feel free to say so in your report if the sentences don't express clear sentiment to you.
# - **Q1.3:** What does the `loop` variable do in the `evaluate()` function?

# %% [markdown]
# # Section 2: Implementing REINFORCE

# %% [markdown]
# ## **Coding Exercises** for Section 2:
# **You will complete the following code blocks denoted by `TODO:`.**
# 
# 

# %% [markdown]
# ### 2.1 Implement the REINFORCE loss

# %%
def compute_reinforce_loss(reward, output_tokens, model):
  """
  Compute REINFORCE loss given the generations' rewards (reward) along with the
  models' generations (output_tokens) that we need to compute probabilities' over.

  Return the log probabilities of the output tokens, and the REINFORCE loss.
  """
  # TODO: get batch_size.
  batch_size = output_tokens.size(0)

  # TODO: feed output_tokens into the model, and get log_probs with output logits
  # using log_softmax.
  output = model(input_ids=output_tokens, attention_mask=(output_tokens != tokenizer.pad_token_id).to(device))
  log_probs = F.log_softmax(output.logits, dim=-1)
  # TODO: choose logprobs of actual generated tokens.
  # Hint: note that output_tokens contains the 10 seed tokens (token 1 ... 10)
  # and the 20 continuation tokens (token 11 ... 30) generated by the model.
  # To get log_probs, putting output_tokens through the model will condition on
  # token 1 ... 30 to predict the next token (at position 31) in the sequence.
  # Therefore, log_probs contains the log probs of tokens at position 2 ... 31.
  # We want to choose the log_probs of tokens at position 2 ... 30 for
  # computing the REINFORCE loss.
  # Hint: chosen_log_probs should have size: torch.Size([batch_size, 29]).
  # Hint: you may find this example code helpful, https://github.com/huggingface/transformers/blob/d90acc16437e8c9e45e068fa1cc1a263b9a7208f/src/transformers/models/gpt2/modeling_gpt2.py#L1103C13-L1103C63
  shifted_logits = output.logits[..., :-1, :].contiguous()
  shifted_labels = output_tokens[..., 1:].contiguous()

  shifted_log_probs = F.log_softmax(shifted_logits, dim=-1)
  gather_indices = shifted_labels[..., 10:].unsqueeze(-1)
  chosen_log_probs = torch.gather(shifted_log_probs, 2, gather_indices).squeeze(-1)

  # TODO: compute batched REINFORCE loss with chosen_log_probs and reward.
  # Hint: it can be helpful to reshape reward for batched computation.
  reward = reward.view(-1, 1)
  batch_loss = (chosen_log_probs * reward).sum(dim=1)

  # TODO: average the batch_loss to get the final REINFORCE loss.
  loss = batch_loss.mean()

  return log_probs, loss

# %% [markdown]
# ### 2.2 REINFORCE training

# %%
######################################################
#  The following helper function is given to you.
######################################################
def reset_model_optimizer(model_name, lr=1e-4):
  global model, optimizer
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  model.config.pad_token_id = tokenizer.eos_token_id
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# %%
def train(model_name, alpha=.001, lr=1e-4, batch_size=32, use_kl_divergence=False, debug=False):
  """
    Train model by computing loss for each mini-batch, and update weights as
    needed using the optimizer.

    For computing the mini-batch loss,
    1. Compute rewards,
    2. Compute REINFORCE loss,
    3. Include KL divergence loss only if use_kl_divergence=True.
  """
  set_seed(42)
  global model, orig_model, tokenizer, reward_model, optimizer
  reset_model_optimizer(model_name, lr=lr)
  global train_texts

  train_texts_ = train_texts[:50] if debug else train_texts

  if use_kl_divergence:
    kl_divergences = []
    orig_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    orig_model.eval()

  losses = []
  rewards = []
  batches = len(train_texts_) // batch_size
  loop = tqdm(total=batches, position=0, leave=False)
  for i in range(0, len(train_texts_), batch_size):
    # TODO: get the current batch of data.
    batch = train_texts_[i:i + batch_size]

    # TODO: compute reward of the batch using compute_reward_for_mini_batch().
    # Hint: you need to detach the reward tensor to break gradient updates.
    reward, output_tokens, _ = compute_reward_for_mini_batch(batch, tokenizer, model, reward_tokenizer, reward_model)
    reward = reward.detach()

    # TODO: compute REINFORCE loss.
    log_probs, loss = compute_reinforce_loss(reward, output_tokens, model)

    # TODO: Compute kl divergence, and modify loss accordingly with alpha.
    # You will complete this for section 3.
    if use_kl_divergence:
      kl_divergence = compute_kl_divergence(log_probs, output_tokens, orig_model)
      kl_divergences.append(kl_divergence.item())

      loss += alpha * kl_divergence

    # TODO: compute gradients and update parameters using optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO: add mean reward, loss, and kl_divergence to their corresponding lists,
    # when appropriate.
    losses.append(loss.item())
    rewards.append(reward.mean().item())

    if use_kl_divergence:
      loop.set_description(f"Loss: {loss.item():.4f}, KL: {kl_divergence.item():.4f}, Reward: {reward.mean().item():.4f}")
    else:
      loop.set_description(f"Loss: {loss.item():.4f}, Reward: {reward.mean().item():.4f}")
    loop.update(1)

  eval_results = evaluate(tokenizer, model, reward_tokenizer, reward_model)

  return_dict = {
      'losses': losses,
      'rewards': rewards,
      'eval_results': eval_results
  }
  if use_kl_divergence:
    return_dict['kl_divergences'] = kl_divergences

  return return_dict

results_reinforce = train(model_name)
print("Losses:", results_reinforce['losses'])

# %% [markdown]
# ### 2.3 Plot

# %%
plt.plot(results_reinforce['rewards'])
plt.xlabel('# batch')
plt.ylabel('reward')

# %% [markdown]
# ## **Questions to Answer** for Section 2:
# **Answer these questions in your write-up report.**
# - **Q2.1:** Run your training function over the entire training set for 1 epoch. Plot the reward over time and report the average reward on the test set of the final trained model. A valid solution should achieve a final average reward of at least 0.75.
#   - *Hint:* An efficient implementation with the recommended hyperparameters above should train the model in under 10 minutes.
# 
# - **Q2.2:** Provide 5 example generations from the model. Are they more positive than the original generations? What else do you notice about them?
# 
# - **Q2.3:** What are the limitations of REINFORCE based on your observations of the generated reviews? Explain what might have caused such limitations.
# 
# - **Q2.4:** What does `reset_model_optimizer()` do?
# 
# - **Q2.5:** What's the shape of `log_probs` in `compute_reinforce_loss()`, and what does each dimension correspond to?

# %% [markdown]
# # Section 3: Regularization

# %% [markdown]
# ## **Coding Exercises** for Section 3:
# **You will complete the following code blocks denoted by `TODO:`.**
# 
# 

# %% [markdown]
# ### 3.1 Compute KL divergence

# %%
def compute_kl_divergence(log_probs_current, output_tokens, orig_model):
  """
  Compute the KL divergence between the original model' log probabilities for generating
  output_tokens, and the current model's log probabilities (log_probs_current)
  """
  # TODO:
  with torch.no_grad():
    orig_logits = orig_model(input_ids=output_tokens, attention_mask=(output_tokens != tokenizer.pad_token_id).to(device)).logits
    orig_probs = F.log_softmax(orig_logits, dim=-1)

  kl_divergence = F.kl_div(log_probs_current, orig_probs, reduction='batchmean', log_target=True)

  return kl_divergence

# %% [markdown]
# ### 3.2 Experiment with different alpha for KL regularization

# %%
results_reinforce_kl_div_alpha_zero = train(model_name, use_kl_divergence=True, alpha=0.000)
plt.plot(results_reinforce_kl_div_alpha_zero['kl_divergences'])
plt.xlabel('# batch')
plt.ylabel('KL')

# %%
results_alpha = train(model_name, use_kl_divergence=True, alpha=0.01)

# %%

plt.plot(results_alpha['kl_divergences'])
plt.xlabel('# batch')
plt.ylabel('KL')

# %%

plt.plot(results_alpha['rewards'])
plt.xlabel('# batch')
plt.ylabel('reward')

# %% [markdown]
# ## **Questions to Answer** for Section 3:
# **Answer these questions in your write-up report.**
# - **Q3.1:** First, run your REINFORCE training function with `alpha=0` for 1 epoch. This is equivalent to training without the KL-penalty. Plot the KL-divergence between the original model and the new model over time. What trend do you see from the plot? Why such a trend could be undesirable in practice?
# - **Q3.2:** Try different `alpha` to experiment with the power of KL-regularization. For each setup, report the `alpha` values, the KL-divergence plots, reward plots, 5 sample generations, and comments on the samples' quality.
#   - A high `alpha` that prevents the model from being able to get a reward of 0.65 or more.
#   - A low `alpha` that the outputs are poor quality.
#   - An `alpha` that gets a good reward (>0.8) and output mostly natural text.
# 
# - **Q3.3:** Based on your experience with REINFORCE with KL-regularization, identify one of its main limitations and propose a potential solution. There's no single right answer here; any justifiable answer will result in full credit.
# 
# - **Q3.4:** Over which tokens do we compute the REINFORCE loss, and why?

# %% [markdown]
# # Section 4: DPO
# 
# In the following section, you will implement training code for DPO. Much of the code can be re-used from the REINFORCE section.
# 
# <!-- Make sure to name your variables and functions **DIFFERENTLY** from the above sections, since you will be re-using functions (like evaluate) and variables from above. -->

# %% [markdown]
# ## **Coding Exercises** for Section 4:
# **You will complete the following code blocks denoted by `TODO:`.**

# %% [markdown]
# ### 4.1 Reloading the base model

# %%
"""
Reload initial model and tokenizer for DPO send it to GPU if available.

This code chunk is the same as in Section 1, so you can copy the code over.

We don't need to reload the reward tokenizer or model!
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', device)
model_name = 'distilgpt2'

# TODO: Load the tokenizer.
# Hint: remember to add paddings on the left for the decoder-only model.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# TODO: set the pad_token to the eos_token.
tokenizer.pad_token = tokenizer.eos_token
# TODO: Load the model and sent it to the device.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.config.pad_token_id = tokenizer.eos_token_id

# %% [markdown]
# ### 4.2 Loading the data
# 
# The previous dataset we used (IMDB) contains only movie reviews and a sentiment label. For DPO, we need to load in a *pairwise, preference* dataset and make a corresponding dataloader.

# %%
"""
Load pairwise IMDB dataset train & test splits, and shuffle the samples within every data split,

We provided a random seed to ensure reproducibility of your shuffling.

Dataset is located at "hallisky/synthetic-imdb-movie-reviews-parallel" on
huggingface datasets.
"""

# TODO: you will store your prompt, concatenated prompt and positive completion,
# and concatenated prompt and negative completion
# Hint: train_texts_parallel and test_texts_parallel are lists of triples of
# (prompt, prompt + pos, prompt + neg).
dataset_parallel = load_dataset('hallisky/synthetic-imdb-movie-reviews-parallel')

train_texts_parallel = [(prompt, prompt_pos, prompt_neg) for prompt, prompt_pos, prompt_neg in zip(
    dataset_parallel['train']['prompt'],
    dataset_parallel['train']['positive_completion'],
    dataset_parallel['train']['negative_completion'])]

test_texts_parallel = [(prompt, prompt_pos, prompt_neg) for prompt, prompt_pos, prompt_neg in zip(
    dataset_parallel['test']['prompt'],
    dataset_parallel['test']['positive_completion'],
    dataset_parallel['test']['negative_completion'])]

random.seed(42)
# TODO: shuffle train and test sets.
random.shuffle(train_texts_parallel)
random.shuffle(test_texts_parallel)

# Print the first 5 rows of data
print(train_texts_parallel[:5])

# %% [markdown]
# ### 4.3 Implementing DPO
# 
# Implement DPO by computing the DPO loss. Note that the prompts are shared between the positive and negative outputs, so we should mask them when we are computing the loss.
# 

# %%
def compute_DPO_loss(policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps,
                     beta = 0.1):
  """
  Compute the DPO loss.

  Args:
    policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
    policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
    reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
    reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
    beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

  Returns:
    The the DPO loss.
    # , the reward of chosen, and reward of rejected.
  """
  # TODO: implement the DPO loss using the equation in the assignment spec.
  # Hint: you may find F.logsigmoid() handy in your implementation.
  # Hint: note the inputs are log probs, so be careful with performing log arithmetic.
  losses = - F.logsigmoid(beta * (policy_chosen_logps - reference_chosen_logps) - beta * (policy_rejected_logps - reference_rejected_logps))

  return losses.mean()

# %%
"""
Helper functions.
"""

def _get_batch_logps(logits: torch.FloatTensor,
                     labels: torch.LongTensor,
                     average_log_prob: bool = False) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    # TODO: choose the per_token_logps for the model generated completion part.
    # Hint: remember to shift logits and labels to make their indices correspond
    # to each other, just like what you did in REINFORCE loss.
    # Hint: you should only use the unmasked tokens (i.e., ignoring the prompts),
    # when you return the final log probs. This can be handled with a loss mask.

    loss_mask = (labels[:, 1:] != -100).float()
    per_token_logps = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def get_log_probs(model, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.FloatTensor:
    """
    Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    # TODO: get the raw output from the model, and get the selected log_probs
    # using the _get_batch_logps() helper function above.
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    all_logps = _get_batch_logps(output.logits, batch['chosen_labels' if 'chosen_labels' in batch else 'rejected_labels'])
    return all_logps


def create_labels_from_attention_mask(tokenized_input, prompt_lengths):
    """
    Creates a labels tensor that matches the size of input_ids. The labels tensor
    is filled with -100 where the attention mask is 0 or at indeces corresponding,
    to the prompt indicating tokens that are NOT using to compute the log probabilties.

    Args:
        tokenized_input (dict): A dictionary containing the tokenized inputs, including 'input_ids' and 'attention_mask'.

    Returns:
        torch.Tensor: A tensor of labels with the same shape as input_ids, filled with -100 where attention_mask is 0.
    """
    # TODO: extract the attention mask from the tokenized input.
    attention_mask = tokenized_input['attention_mask']

    # TODO: get the length difference between the full input_lengths and the prompt_lengths.
    input_lengths = attention_mask.sum(dim=1)
    length_diff = input_lengths - prompt_lengths

    # TODO: create a labels tensor that matches the shape of input_ids, initially filled with -100.
    labels = torch.full_like(tokenized_input['input_ids'], fill_value=-100).to(device)

    # TODO: update labels to match input_ids values where the attention mask is 1 (keeping -100 where it is 0).
    labels = torch.where(attention_mask == 1, tokenized_input['input_ids'], labels)

    # TODO: update labels so that prompts from the input_ids are set to -100
    labels = torch.where(length_diff.unsqueeze(1) > 0, labels, -100)

    return labels


# %% [markdown]
# ### 4.4 Training DPO

# %%
def train_DPO(model_name, beta=0.5, lr=5e-7, batch_size=32, max_length=128, debug=False):
  """
    Trains a specified model using Deep Policy Optimization (DPO) by computing
    loss for each mini-batch, and update weights as needed using the optimizer.

    Parameters:
    - model_name (str): Name of the model to be trained. This is a required parameter that specifies
      which model architecture to load and train.
    - beta (float, optional): Regularization parameter for DPO
    - lr (float, optional): Learning rate for the optimizer.
    - batch_size (int, optional): Size of the batches of data used for each iteration of training.
    - debug (bool, optional): Enables or disables debug mode
    - max_length: Max length of sequences in batch
  """
  set_seed(42)
  global model, orig_model, tokenizer, reward_model, optimizer
  reset_model_optimizer(model_name, lr=lr)
  global train_texts

  train_texts_ = train_texts_parallel[:50] if debug else train_texts_parallel

  # TODO: save the original model.
  orig_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

  losses = []
  chosen_rewards = []

  batches = len(train_texts_) // batch_size
  loop = tqdm(total=batches * 2, position=0, leave=False)
  for _ in range(2):
    for i in range(0, len(train_texts_), batch_size):
      # TODO: get the current batch of data.
      batch = train_texts_[i:i + batch_size]

      """
      Convert the batch to tokenized form separately for the prompt, the positive, and negative examples.

      Remember to send all tokenized data to the device.
      """
      # TODO: tokenize the prompt and get the prompt_lengths, and set padding=True,
      # truncation=True, and max_length=max_length.
      prompt_tokenized = tokenizer([x[0] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
      prompt_lengths = prompt_tokenized['attention_mask'].sum(dim=1).to(device)

      # TODO: tokenize the chosen sequence.
      chosen_tokenized = tokenizer([x[1] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

      # send chosen_tokenized to the device.
      chosen_tokenized = {k: v.to(device) for k, v in chosen_tokenized.items()}

      # TODO: create labels from attention_mask for chosen_tokenized.
      chosen_tokenized['chosen_labels'] = create_labels_from_attention_mask(chosen_tokenized, prompt_lengths)

      # TODO: tokenize the rejected sequence.
      rejected_tokenized = tokenizer([x[2] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

      # send rejected_tokenized to the device.
      rejected_tokenized = {k: v.to(device) for k, v in rejected_tokenized.items()}

      # TODO: create labels from attention_mask for rejected_tokenized.
      rejected_tokenized['rejected_labels'] = create_labels_from_attention_mask(rejected_tokenized, prompt_lengths)

      # TODO: get log probs for the chosen and rejected model completions from the policy model.
      model_chosen_log_probs = get_log_probs(model, chosen_tokenized)
      model_rejected_log_probs = get_log_probs(model, rejected_tokenized)

      # TODO: get log probs for the chosen and rejected model completions from the original model.
      with torch.no_grad():
        orig_chosen_log_probs = get_log_probs(orig_model, chosen_tokenized)
        orig_rejected_log_probs = get_log_probs(orig_model, rejected_tokenized)

      # TODO: compute DPO loss.
      loss = compute_DPO_loss(model_chosen_log_probs, model_rejected_log_probs, orig_chosen_log_probs, orig_rejected_log_probs, beta=beta)

      # TODO: compute gradients and update parameters using optimizer.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # TODO: add loss to the list.
      losses.append(loss.item())

      loop.set_description(f"Loss: {loss.item():.4f}") # , Reward: {reward.mean().item():.4f}")
      loop.update(1)

  eval_results = evaluate(tokenizer, model, reward_tokenizer, reward_model)

  # TODO: repeat the above logic to get the loss on the test set after training.
  test_loss = 0
  test_batches = len(test_texts_parallel) // batch_size
  loop = tqdm(total=test_batches, position=0, leave=False)

  for i in range(0, len(test_texts_parallel), batch_size):
    batch = test_texts_parallel[i:i + batch_size]

    prompt_tokenized = tokenizer([x[0] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_lengths = prompt_tokenized['attention_mask'].sum(dim=1).to(device)

    chosen_tokenized = tokenizer([x[1] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    chosen_tokenized = {k: v.to(device) for k, v in chosen_tokenized.items()}
    chosen_tokenized['chosen_labels'] = create_labels_from_attention_mask(chosen_tokenized, prompt_lengths)

    rejected_tokenized = tokenizer([x[2] for x in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    rejected_tokenized = {k: v.to(device) for k, v in rejected_tokenized.items()}
    rejected_tokenized['rejected_labels'] = create_labels_from_attention_mask(rejected_tokenized, prompt_lengths)

    model_chosen_log_probs = get_log_probs(model, chosen_tokenized)
    model_rejected_log_probs = get_log_probs(model, rejected_tokenized)

    with torch.no_grad():
      orig_chosen_log_probs = get_log_probs(orig_model, chosen_tokenized)
      orig_rejected_log_probs = get_log_probs(orig_model, rejected_tokenized)

    tloss = compute_DPO_loss(model_chosen_log_probs, model_rejected_log_probs, orig_chosen_log_probs, orig_rejected_log_probs, beta=beta)

    test_loss += tloss.item()

    loop.set_description(f"Test Loss: {test_loss:.4f}")
    loop.update(1)

  test_loss /= test_batches

  return_dict = {
      'losses': losses,
      'test_loss': test_loss,
      'eval_results': eval_results
  }

  return return_dict

results_DPO = train_DPO(model_name, batch_size=16, debug=False, beta=0.175)

# %% [markdown]
# ### 4.5 Plots

# %%
plt.plot(results_DPO['losses'])
plt.xlabel('# batch')
plt.ylabel('train loss')
print("Test Loss:", results_DPO['test_loss'])

# %%
tokenizer.batch_decode(model.generate(tokenizer(["I expected a great"], return_tensors="pt")["input_ids"].to('mps')))

# %% [markdown]
# ## **Questions to Answer** for Section 4:
# **Answer these questions in your write-up report.**
# - **Q4.1:** Run your training function for 2 epochs using the batch size 16. Plot the training/test losses over time and report the model's average loss on the test set for the final model. Note that an efficient implementation should run in about 10 minutes.
# 
# - **Q4.2:** Run the evaluation function using the same data and code from Part 1 using the DPO-trained model. How does the reward compare to the REINFORCE model? Why do you think the results look like this?
# 
# - **Q4.3:** Provide 5 example generations from the DPO-trained model. Are they more positive than the generations of the original model? What else do you notice about them?
# 
# - **Q4.4:** Try 3 different values for `β` (note it's typically set to be in the range of 0.1 to 0.5). Plot the training loss plots and report the final test losses for each of these setups. What's the best setup that you found? How do different values of `β` impact the training differently?
# 
# - **Q4.5:** What's the purpose of using `σ`, the logistic function, in `L_DPO`?
# 
# - **Q4.6:** What are the main differences and advantages of DPO compared to other online RL methods like PPO?


