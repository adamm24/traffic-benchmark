from __future__ import annotations

import platform


class HuggingFaceBackend:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str | None = None,
        max_new_tokens: int = 16,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face backend needs torch and transformers installed."
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = _select_device(device, torch)
        print(f"Selected device: {self.device}")
        if self.device == "mps" and platform.system() == "Darwin":
            print("Warning: MPS selected on macOS. Use --device cpu if PyTorch MPS fails.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        kwargs = {"trust_remote_code": trust_remote_code}
        if dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        elif dtype:
            kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded: {model_id}")

    def generate(self, prompt: str) -> str:
        text = self._chat_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _chat_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt


def _select_device(requested: str, torch_module) -> str:
    if requested == "auto":
        return _best_device(torch_module)
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested == "mps" and not _mps_available(torch_module):
        raise RuntimeError("MPS requested but not available.")
    return requested


def _best_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if _mps_available(torch_module):
        return "mps"
    return "cpu"


def _mps_available(torch_module) -> bool:
    return hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
