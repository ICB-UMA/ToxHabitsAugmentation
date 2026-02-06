from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

class SimpleNERModel(torch.nn.Module):
    def __init__(self, model_name, num_tag):
        super().__init__()

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self.num_tag = num_tag
        self.parent_model = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        self.drop = torch.nn.Dropout(0.3)
        self.output = torch.nn.Linear(self.parent_model.config.hidden_size, self.num_tag)
        torch.nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            torch.nn.init.zeros_(self.output.bias)

    def get_max_len(self, max_len: int = 256) -> int:
        return max_len - self.tokenizer.num_special_tokens_to_add(pair=False)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.parent_model(input_ids, attention_mask=attention_mask, return_dict=False)[0]
        logits = self.output(self.drop(output))
        return logits
