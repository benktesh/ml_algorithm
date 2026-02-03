# Fine-Tuning Techniques

## Standard Fine-Tuning
Adjusts all the parameters in LLM to increase performance for a specific task. Extremely effective, but also expensive.

## Low-Rank Adaptation (LoRA)
Parameter-efficient fine-tuning. Instead of being intrusive (updating all weights), it only adds small, low-rank matrices to specific layers — reducing the number of parameters that need to be updated.

## Supervised Fine-Tuning (SFT)
Trains a base model on a new dataset under supervision (including demonstration data, prompt-response pairs).

## Reinforcement Learning from Human Feedback (RLHF)
Trains models with the help of humans — facilitates continuous improvement based on human input.

# Fine-Tuning Process

Data/datasets --> Prepare training data --> Add task-specific head (new layers) --> Train the model -->evaluate Model --> deploy the model
