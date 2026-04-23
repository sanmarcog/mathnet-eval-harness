import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

print("Attaching LoRA...")
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Building toy dataset...")
toy_data = [{"text": f"Q: What is {i}+{i}?\nA: {2*i}"} for i in range(1, 11)]
dataset = Dataset.from_list(toy_data)

print("Training 10 steps...")
training_args = SFTConfig(
    output_dir="./toy_adapter",
    max_steps=10,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=10,
    bf16=True,
    report_to="none", dataset_text_field="text",
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./toy_adapter")
print("Done.")
