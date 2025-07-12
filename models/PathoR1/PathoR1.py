from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class PathoR1:
    def __init__(self, model_path: str, args):
        """
        Initialize the PathoR1 multimodal pathology model and processor.

        Args:
            model_path: Hugging Face model ID or local path.
            args: Argument object with attributes:
                - max_new_tokens: int
                - temperature: float
                - top_p: float
                - repetition_penalty: float
        """
        # Load model and processor
        super().__init__()
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Generation parameters
        self.generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty
        }

    def process_messages(self, messages: list) -> dict:
        """
        Prepare prompt and vision inputs.

        Args:
            messages: list of dicts; each dict contains 'role' and 'content'.

        Returns:
            dict of inputs for generation: tokenized text and vision tensors.
        """
        # 1. Build text prompt
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 2. Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)
        # 3. Tokenize and collate
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # 4. Move to model device
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        return inputs

    def generate_output(self, messages: list) -> str:
        """
        Generate a response for a single input.

        Args:
            messages: list of message dicts.

        Returns:
            Generated response string.
        """
        inputs = self.process_messages(messages)
        # 5. Inference
        output_ids = self.llm.generate(
            **inputs,
            **self.generation_config
        )
        # 6. Trim prompt tokens
        trimmed = []
        for inp_ids, out_ids in zip(inputs['input_ids'], output_ids):
            trimmed.append(out_ids[inp_ids.size(0):])
        # 7. Decode outputs
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return texts[0]

    def generate_outputs(self, messages_list: list) -> list:
        """
        Batch generate for multiple inputs.

        Args:
            messages_list: list of messages lists.

        Returns:
            List of response strings.
        """
        return [self.generate_output(msgs) for msgs in messages_list]
