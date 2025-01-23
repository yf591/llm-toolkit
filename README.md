# llm-toolkit

A collection of tools and recipes for working with Large Language Models (LLMs). This repository provides various utilities and examples for tasks such as:

*   Text Generation
*   Supervised Fine-Tuning (SFT) (Potentially)
*   Reinforcement Learning from Human Feedback (RLHF) (Potentially)
*   Direct Preference Optimization (DPO) (Potentially)

## Getting Started

To get started with this toolkit, follow these steps:

1.  **Open in Colab:** Click the "Open in Colab" button below to open the notebook in Google Colab.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)

2.  **Set Runtime Type:** In Colab, go to "Runtime" -> "Change runtime type" and select "GPU" as the hardware accelerator. This is highly recommended for faster performance, especially with larger models.

3.  **Install Libraries:** Run the following command in a Colab code cell to install the necessary libraries:

    ```bash
    !pip install transformers torch gradio
    ```

4.  **Mount Google Drive (Optional):** If you need to access files in your Google Drive, run the following code in a Colab code cell and follow the authentication prompts:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

5.  **Run the Notebook:** Execute the code cells in the notebook.

## Examples

### Text Generation

The following code demonstrates basic text generation using a pre-trained language model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Model and tokenizer loading
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def generate_text(prompt, max_input_length=1024, max_output_length=200):
    inputs = tokenizer(prompt, truncation=True, max_length=max_input_length, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", lines=10, max_lines=20),
        gr.Slider(minimum=128, maximum=2048, value=1024, label="Max Input Length"),
        gr.Slider(minimum=50, maximum=500, value=200, label="Max Output Length")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10, max_lines=20)
)

interface.launch(share=True)
```

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yf591/llm-toolkit/blob/main/LICENSE) file for details.
