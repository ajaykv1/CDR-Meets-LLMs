import os
import wandb
import torch 
import transformers
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset,DatasetDict
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments


def load_train_val_data(k_shot, source, target, data_info, rating_ranking, injection, prompt_context):
    train_df = pd.read_csv(f"/scratch/akrish/fall_2023/src/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_{rating_ranking}_{injection}_injection.csv")
    validation_df = pd.read_csv(f"/scratch/akrish/fall_2023/src/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_{rating_ranking}_{injection}_injection.csv")
    test_df = pd.read_csv(f"/scratch/akrish/fall_2023/src/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_{rating_ranking}_{injection}_injection.csv")
    
    print("Loaded Data\n\n")

    # train_df = train_df.head(10)  
    # validation_df = validation_df.head(10)   
    # test_df = test_df.head(1)   

    if rating_ranking == 'rating':
        # train_df['text'] =  train_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['ground_truth']), axis = 1)
        # validation_df['text'] =  validation_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['ground_truth']), axis = 1)
        # test_df['text'] =  validation_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['ground_truth']), axis = 1)

        system_prompt = ""

        if prompt_context == 'none':
            system_prompt = "<s> [INST] <<SYS>>You will output a number.<</SYS>>"

        elif prompt_context == 'medium':
            system_prompt = "<s> [INST] <<SYS>>You will output a single floating point number, which will represent the rating a user will give to an item in the target domain. <</SYS>>"

        elif prompt_context == 'high':
            system_prompt = "<s> [INST] <<SYS>>You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. You will output a single floating point number, which will represent the rating a user will give to an item in the target domain. <</SYS>>"


        train_df['text'] =  train_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['ground_truth']) + " </s>", axis = 1)
        validation_df['text'] =  validation_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['ground_truth']) + " </s>", axis = 1)
        test_df['text'] =  validation_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['ground_truth']) + " </s>", axis = 1)

    if rating_ranking == 'ranking':

        # train_df['text'] =  train_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['correct_ranking']), axis = 1)
        # validation_df['text'] =  validation_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['correct_ranking']), axis = 1)
        # test_df['text'] =  validation_df.apply(lambda row: "### Input: " + str(row['prompt']) + " ### Output: " + str(row['correct_ranking']), axis = 1)

        system_prompt = ""

        if prompt_context == 'none':
            system_prompt = "<s> [INST] <<SYS>>You will output a list<</SYS>>"

        elif prompt_context == 'medium':
            system_prompt = "<s> [INST] <<SYS>>You will output a ranked list, where items will be ranked from most likely to interact with to least likely to interact with. <</SYS>>"

        elif prompt_context == 'high':
            system_prompt = "<s> [INST] <<SYS>>You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. You will output a ranked list of candidate items. The first item in the list shohuld be the item that the user will most likeley interact with, and the last item in the list should be the item the user will be least likely to interact with. You should follow the expected format in the prompt. <</SYS>>"

        train_df['text'] =  train_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['correct_ranking']) + " </s>", axis = 1)
        validation_df['text'] =  validation_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['correct_ranking']) + " </s>", axis = 1)
        test_df['text'] =  validation_df.apply(lambda row: str(system_prompt) + str(row['prompt']) + " [/INST] " + str(row['correct_ranking']) + " </s>", axis = 1)

    print(train_df) 

    fineune_dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(validation_df),
        "test": Dataset.from_pandas(test_df)
    })

    return fineune_dataset_dict

def load_model_and_tokenizer(model_name):
    # This is the non-chat version
    # llm_name = f"/scratch/akrish/fall_2023/src/LLMs/llama/llama-2-{model_name.lower()}/{model_name}"
    
    # This is the chat version
    llm_name = f"/scratch/akrish/fall_2023/src/LLMs/llama/llama-2-{model_name.lower()}-chat/{model_name}"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map = "auto"
    )

    model.config.pretraining_tp = 1
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer

def load_peft_configuration ():

    peft_config = LoraConfig(
        lora_alpha= 16,
        lora_dropout= 0.1, 
        r= 64, 
        bias="none",
        task_type="CAUSAL_LM"
    )

    return peft_config

def load_training_arguments(k_shot, target, model_name, rating_ranking, injection, prompt_context):
    run_name = f"{target}_{model_name}_{rating_ranking}_{injection}_injection_{k_shot}_{prompt_context}"
    output_dir = f"./results/{prompt_context}/{target}_{model_name}_{rating_ranking}_{injection}_injection_{k_shot}"
    per_device_train_batch_size = 4 
    per_device_eval_batch_size = 4 
    gradient_accumulation_steps = 1
    eval_accumulation_steps = 1
    optim = "paged_adamw_32bit"
    save_steps = 2000 
    logging_steps = 1
    learning_rate = 0.00002
    fp16 = True
    max_grad_norm = 0.3
    max_steps = 10000 
    warmup_ratio = 0.03
    group_by_length = True
    lr_scheduler_type = "constant"
    evaluation_strategy = "steps"
    eval_steps = 2000 

    training_arguments = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps = eval_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy=evaluation_strategy,
        eval_steps= eval_steps
    )

    return training_arguments

def fine_tune_llm(k_shot, source, target, data_info, model_name, rating_ranking, injection, prompt_context):

    os.environ["WANDB_PROJECT"]="llama-2-chat-cdr"
    #os.environ["WANDB_PROJECT"]="llama-2-fine-tuning" #"llm_project"
    #os.environ["WANDB_API_KEY"] = "b2a2d27f3016010cfd244efb0de120c2392c5d5b"
    #os.environ["WANDB_MODE"] = "offline"

    # Checking GPU availability
    print("\n\nIS GPU AVAILABLE: " + str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    print("Loading model and tokenizer.....")
    fineune_dataset_dict = load_train_val_data(k_shot, source, target, data_info, rating_ranking, injection, prompt_context)
    model, tokenizer = load_model_and_tokenizer(model_name)

    print("\n\nLoading PEFT Config and peparing model for training.....")
    peft_config = load_peft_configuration()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    print("Loading training arguments.....")
    training_arguments = load_training_arguments(k_shot, target, model_name, rating_ranking, injection, prompt_context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
        print("NUMBER OF GPUs: " + str(torch.cuda.device_count()))

    max_seq_length = 4096

    # setting resonse template for LLM
    response_template = "[/INST]" 
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    print("Loading SFTTrainer.....")
    trainer = SFTTrainer(
        model=model,
        train_dataset=fineune_dataset_dict['train'],
        eval_dataset=fineune_dataset_dict['validation'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_arguments
    )
    
   
    print("Start Fine-Tuning LLM.....\n\n")
    trainer.train()

    print("\n\nTraining Finished.....")
    wandb.finish()

    print("Saving Peft Model Adapters.....\n\n")
    trainer.save_model(f"/scratch/akrish/fall_2023/src/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/fine_tuned_llms/{target}_{model_name}_{rating_ranking}_{injection}_injection")

    base_model_id = f"/scratch/akrish/fall_2023/src/LLMs/llama/llama-2-{model_name.lower()}-chat/{model_name}"
    peft_model_id = f"/scratch/akrish/fall_2023/src/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/fine_tuned_llms/{target}_{model_name}_{rating_ranking}_{injection}_injection"
    # base_model_id = f"/scratch/akrish/fall_2023/src/LLMs/llama/llama-2-{model_name.lower()}/{model_name}"

    b_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            return_dict=True,
            load_in_4bit=True,
            device_map="auto",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            return_dict=True,
            load_in_4bit=True,
            device_map="auto",
        )
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id
    )
    trained_model = PeftModel.from_pretrained(
        b_model,
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )

    trained_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id
    )

    
    test_dataframe = fineune_dataset_dict['test'].to_pandas().head(1)
    # Extract the full string from the first row of the 'text' column
    full_texts = test_dataframe['prompt'].values
    full_string = full_texts[0]

    # Split the string into two parts
    # split_string = full_string.split("[/INST] ")
    # The first part contains the input and the label "### Output: "
    input_with_label = "[INST] " + str(full_string) + " [/INST] " # split_string[0] + "[/INST] "

    # The second part is the ground truth
    # ground_truth = split_string[1] if len(split_string) > 1 else ""

    ground_truth = ""

    if rating_ranking == 'rating':
        ground_truths = test_dataframe['ground_truth'].values
        ground_truth = ground_truths[0]
    elif rating_ranking == 'ranking':
        ground_truths = test_dataframe['correct_ranking'].values
        ground_truth = ground_truths[0]

    print("\nnput is: \n\n" + str(input_with_label))
    print("\n\nGround truth is: \n\n" + str(ground_truth))

    input_ids = trained_tokenizer(input_with_label, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=100)
    output = trained_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_with_label):]

    print("\n\n\nFINE_TUNED PREDICTION IS: ")
    print(output)

    input_ids = base_tokenizer(input_with_label, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = base_model.generate(input_ids=input_ids, max_new_tokens=100)
    output = base_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_with_label):]

    print("\n\n\nBase PREDICTION IS: ")
    print(output)











    