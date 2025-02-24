import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv('.env')

class FineTuningTask(Task):
    def __init__(self, model_name, dataset_name, output_dir, llm_provider):
        super().__init__(
            description="Fine-tuning strategy",
            goal="Train an LLM and suggest best hyperparameters",
            output_format="JSON format with training settings"
        )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.llm = llm_provider

    def execute(self) -> TaskResult:
        try:
            prompt = f"""
            You are an expert in training language models.
            The model {self.model_name} will be fine-tuned using the dataset {self.dataset_name}.

            Requirements:

            Suggest the best hyperparameters (learning rate, batch size, warmup steps).
            Indicate if LoRA or QLoRA should be used to optimize memory.
            Report any potential risks or issues based on the dataset.
            Return a JSON in the format:

            {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "warmup_steps": 100,
                "use_LoRA": true,
                "recommendations": "Avoid overfitting by using early stopping."
            }
            """
            hyperparams = self.llm.generate(prompt)
            
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dataset = load_dataset(self.dataset_name)

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=hyperparams.get("batch_size", 4),
                num_train_epochs=1,
                weight_decay=0.01
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"]
            )
            
            trainer.train()
            model.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            
            return TaskResult(success=True, content=self.output_dir)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class ModelDeploymentTask(Task):
    def __init__(self, model_dir, llm_provider):
        super().__init__(
            description="Model Deployment Strategy",
            goal="Deploy a trained LLM via API",
            output_format="JSON format with deployment recommendations"
        )
        self.model_dir = model_dir
        self.llm = llm_provider

    def execute(self) -> TaskResult:
        try:
            prompt = f"""
            You are an expert in deploying language models.
            The trained model is located in {self.model_dir}.

            Tasks:

            - Suggest the best architecture to serve this model (FastAPI, Flask, Triton, vLLM, etc.).
            - Provide the minimum required resources (RAM, VRAM, CPU).
            - Generate a code snippet to load the model and create a /generate endpoint.            
            """
            deployment_recommendations = self.llm.generate(prompt)
            
            model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            app = FastAPI()

            @app.post("/generate")
            async def generate_text(prompt: str):
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=200)
                return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
            
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
            
            return TaskResult(success=True, content="http://localhost:8000/generate")
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


async def main():
    model_name = "distilgpt2"
    dataset_name = "wikitext"
    output_dir = "./trained_model"
    llm_provider = GeminiProvider(GeminiConfig())
    
    fine_tuning_task = FineTuningTask(model_name, dataset_name, output_dir, llm_provider)
    deploy_task = ModelDeploymentTask(output_dir, llm_provider)

    fine_tuning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[fine_tuning_task]
    )
    
    deploy_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[deploy_task]
    )
    
    orchestrator = Orchestrator(
        agents={"fine_tuning": fine_tuning_agent, "deploy": deploy_agent},
        flows={
            "fine_tune_and_deploy": Flow(agents=["fine_tuning", "deploy"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    result = await orchestrator.process_request(None, flow_name="fine_tune_and_deploy")
    
    if result["success"]:
        print("\nDeployment successful! API running at:")
        print(result["content"])
    else:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
