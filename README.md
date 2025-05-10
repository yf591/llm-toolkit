# LLM Toolkit 🛠️

This repository serves as a toolkit consolidating sample code and learning resources that I have created and practiced for learning Large Language Models (LLMs). It covers a range of techniques for deeply leveraging LLMs, from the basics of text generation to advanced applications like Supervised Fine-Tuning (SFT) and Retrieval Augmented Generation (RAG), all presented in Google Colab notebook format.

My goal is not only to enhance my own knowledge and skills in the LLM field but also to provide a helpful starting point for others learning in this area to begin their development journey. I've made this public with the hope that it can become step-by-step learning content, even for beginners.

## 🚀 Featured Notebooks

| Notebook                                                                                                                               | Description                                                                                                                                                           | Key Technologies/Libraries                                                                                                         | Features & What You'll Learn                                                                                             | Open in Colab                                                                                                                                    |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Universal LLM GUI Notebook](https://github.com/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)**                      | A versatile Gradio UI-equipped notebook allowing you to select various pre-trained LLMs and experiment with text generation by adjusting parameters.                  | `transformers`, `torch`, `gradio`                                                                                              | Basic LLM operations, model selection, parameter tuning, easy text generation experience                                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb) |
| **[SFT LLM FineTuning GUI Notebook](https://github.com/yf591/llm-toolkit/blob/main/SFT_LLM_FineTuning_GUI_Notebook.ipynb)**               | Explains how to efficiently fine-tune LLMs using QLoRA with the `SFTTrainer` from the `trl` library. Allows testing the fine-tuned model via a Gradio UI.        | `transformers`, `peft` (QLoRA), `trl` (SFTTrainer), `bitsandbytes`, `datasets`, `gradio`, `torch`, `accelerate`                  | SFT (QLoRA), `SFTTrainer` usage, dataset formatting, efficient fine-tuning, Gradio UI for evaluation                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/SFT_LLM_FineTuning_GUI_Notebook.ipynb) |
| **[LLM SFT & RAG with GradioUI](https://github.com/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb)**                        | Builds a chatbot that responds based on specific domain knowledge by combining QLoRA-based SFT with RAG. Test it interactively with a Gradio UI.                  | `transformers`, `peft` (QLoRA), `langchain`, `faiss-cpu`, `bitsandbytes`, `datasets`, `gradio`, `torch`, `accelerate`                | SFT (QLoRA) + RAG, domain knowledge integration, vector DB (FAISS), LangChain pipelines, interactive UI, model sharing          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb) |
| *RLHF_Notebook.ipynb (Placeholder)*                                                                                                     | *(In Preparation)* A notebook for practicing Reinforcement Learning from Human Feedback (RLHF).                                                              | `trl` (planned), `transformers`, `datasets`                                                                                       | *(Planned)* Reward modeling, PPO algorithm, LLM behavior alignment                                                                | *(Coming Soon)*                                                                                                                            |
| *DPO_Notebook.ipynb (Placeholder)*                                                                                                      | *(In Preparation)* A notebook for learning Direct Preference Optimization (DPO), a simpler alignment method.                                                       | `trl` (planned), `transformers`, `datasets`                                                                                       | *(Planned)* Preference pair learning, RLHF alternative                                                                             | *(Coming Soon)*                                                                                                                            |

## 📚 Key Technologies You Can Learn & Experiment With

Through this toolkit, you can deepen your understanding and acquire practical skills in the following LLM-related technologies

*   **LLM Basics and Inference**: The fundamental workflow of loading various models from Hugging Face Hub and generating text.
*   **Efficient Fine-Tuning (Supervised Fine-Tuning: SFT)**
    *   **QLoRA (Quantized Low-Rank Adaptation)**: A technique for efficiently fine-tuning large models in low-resource environments.
    *   **Comparative Learning of Different Implementation Approaches**
        *   **`SFTTrainer` from `trl` library** (used in `SFT_LLM_FineTuning_GUI_Notebook.ipynb`)
            *   **Strengths**: A tool for achieving high performance **easily and efficiently** for **standard SFT use cases** (especially dialogue and instruction formats). Leads to concise code and readily handles SFT-specific data processing (like packing).
            *   **Considerations**: May have limitations for highly specialized SFT tasks deviating from standards, or for very detailed customization of the training loop.
        *   **Custom PyTorch Training Loop** (used in `LLM_SFT_&_RAG_GradioUI.ipynb`)
            *   **Strengths**: Offers **maximum flexibility and control**. Allows implementation of any kind of data format (e.g., RAG-augmented prompts), training logic, loss functions, etc. Enables pursuit of **potential peak performance** through highly tailored tuning for specific unique requirements.
            *   **Considerations**: Increases implementation complexity, requiring more knowledge and experimentation. Convenient features provided by `SFTTrainer` need to be implemented manually.
    *   Adapting models to specific tasks or custom datasets (e.g., instruction tuning, dialogue tuning, RAG-augmented tuning).
*   **RAG (Retrieval Augmented Generation)**
    *   Methods for leveraging external documents (PDFs, text files, etc.) as a real-time knowledge source for LLMs.
    *   **Vector Databases (FAISS)**: Vectorizing documents and enabling fast similarity searches.
    *   **LangChain**: Efficiently building and managing RAG pipelines (document loading, splitting, embedding, retrieval, prompt generation, LLM integration).
*   **Interactive UI Development**: Creating simple web UIs for easy model testing using Gradio.
*   **Hugging Face Hub Integration**: Sharing your trained models (adapters) on the Hub.

## 🛠️ Basic Notebook Usage

1.  **Open Notebook in Colab**
    *   Click the "Open in Colab" badge in the table above or directly open the notebook file link.
2.  **Configure Runtime (Important)**
    *   From the Colab menu, select "Runtime" → "Change runtime type".
    *   Under "Hardware accelerator," select **"GPU"** (e.g., T4, L4, A100, whichever is available). A GPU is essential, especially for fine-tuning and running large models. The free tier of Colab may offer T4 GPUs.
3.  **Run Setup Cells**
    *   The initial cells in each notebook contain code for installing necessary libraries and setting up the environment. Execute these cells sequentially.
    *   **Note**: Notebooks involving SFT or RAG install many libraries, so the initial setup might take a few minutes.
4.  **Prepare Data (As Needed by Notebook)**
    *   For fine-tuning notebooks like `SFT_LLM_FineTuning_GUI_Notebook.ipynb` or `LLM_SFT_&_RAG_GradioUI.ipynb`, you'll need to upload training datasets or RAG document files to your Colab environment or place them in specific paths. Follow the instructions within each notebook.
5.  **Execute Cells Sequentially**
    *   Run each cell from top to bottom, reading the explanations and instructions provided in the Markdown cells.
    *   When you run cells that launch a Gradio UI, an interactive web interface will appear in the Colab output area, allowing you to test the model.

## 📖 Notebook Highlights

### 1. Universal LLM GUI Notebook 💬
A versatile notebook for easily loading and experimenting with various pre-trained LLMs (e.g., `elyza/ELYZA-japanese-Llama-2-7b-instruct`, `google/gemma-7b-it`). It features a Gradio-based UI for prompt input and parameter adjustment (temperature, max new tokens, etc.).

**Recommended for users who want to**
*   Quickly try out different LLMs.
*   Practice prompt engineering.
*   Learn basic UI construction with Gradio.

*(Code examples within this notebook are designed to be generic to support various models.)*

### 2. SFT LLM FineTuning GUI Notebook 🎯
This notebook provides a step-by-step guide on efficiently performing Supervised Fine-Tuning (SFT) on Large Language Models (e.g., `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1`) using QLoRA (Quantized Low-Rank Adaptation) and the `SFTTrainer` from Hugging Face's `trl` library. It covers the entire process from custom dataset preparation, training configuration, and execution, to verifying the fine-tuned model, concluding with an easy-to-use Gradio UI for inference.
*Note: While `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1` does not require Hugging Face login for access, some other models might.*

**Key Steps and Learning Points**
1.  **Environment Setup**: GPU check, library installation, Hugging Face Hub login.
2.  **Model & Tokenizer Preparation**: Loading the base model with 4-bit quantization for QLoRA.
3.  **Dataset Preparation**: Loading a custom dataset (instruction-response pairs) and formatting it into the chat template required by `SFTTrainer`.
4.  **QLoRA & Training Configuration**: Identifying LoRA target modules, setting up `LoraConfig` and `TrainingArguments`.
5.  **Fine-tuning with `SFTTrainer`**: Executing efficient SFT with just a few lines of code.
6.  **Result Verification & Model Saving**: Saving the trained LoRA adapter and checking the number of trainable parameters.
7.  **Inference with Fine-tuned Model**: Applying the saved adapter to the base model and interactively evaluating its performance using a Gradio UI.

**Recommended for users who want to**
*   Fine-tune an LLM on a specific instruction-response dataset.
*   Master **easy and efficient** training methods like QLoRA using **`SFTTrainer`**.
*   Try customizing Japanese LLMs.
*   Experiment with SFT on limited computational resources.

### 3. LLM SFT & RAG with GradioUI 🧠📚
This notebook extends the SFT techniques by integrating Retrieval Augmented Generation (RAG) to dynamically incorporate external specialized knowledge into an LLM. It demonstrates building a chatbot using a Japanese LLM (e.g., `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1`) that leverages a specific document (e.g., "Reinforcement Learning: An Introduction" PDF) as its knowledge source.
*This also serves as an experiment to observe the performance when implementing RAG with English documents for a Japanese LLM.*

**Key Steps and Learning Points**
1.  **RAG System Construction**: Build a vector database (FAISS) using your chosen specialized document as the knowledge source. Covers document loading, chunking, embedding, and index creation.
2.  **SFT Data Preparation**: Augment an instruction-response dataset (e.g., `ichikara-instruction`) by adding relevant context retrieved by the RAG system for each question. This trains the model to "read and answer based on context."
3.  **SFT with QLoRA**: Fine-tune the base model using the RAG-augmented dataset with memory-efficient QLoRA. This notebook utilizes a **custom PyTorch training loop** (or the Hugging Face Trainer), allowing you to learn how to flexibly control the data format and training process tailored for RAG.
4.  **Inference Pipeline Construction**: Integrate the fine-tuned LoRA adapter and the RAG system using LangChain to build a question-answering pipeline.
5.  **Interactive Demo with Gradio UI**: Qualitatively evaluate the performance of the created RAG chatbot through an interactive Gradio interface.
6.  **Sharing to Hugging Face Hub**: Steps to upload the trained LoRA adapter to the Hub.

**Recommended for users who want to**
*   Build an LLM that can respond based on specific domain knowledge.
*   Learn advanced LLM customization techniques combining SFT and RAG, and **are interested in customizing the training loop**.
*   Understand the implementation of RAG using LangChain and vector databases.
*   Get a practical recipe for developing LLM applications.

## 💡 Future Roadmap

*   **RLHF (Reinforcement Learning from Human Feedback) Notebook**: For training LLMs to generate more natural responses aligned with user intent.
*   **DPO (Direct Preference Optimization) Notebook**: A simpler alternative to RLHF for fine-tuning LLMs directly from preference data.
*   Support for various model sizes (from lightweight to large-scale).
*   Expansion of sample notebooks for diverse tasks (summarization, translation, code generation, etc.).
*   Enrichment of explanatory documentation for each notebook (theoretical background, implementation details).
*   Introduction of specific metrics and methods for model evaluation.

## 🙏 Contributions

Contributions in any form are highly welcome, whether it's bug reports, feature suggestions, new notebook additions, or documentation improvements! Please feel free to create an Issue or send a Pull Request on GitHub. Let's make this toolkit even better together.

## 📜 Disclaimer

The software and sample code provided in this repository are offered "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
The content generated by Large Language Models (LLMs) may contain inaccuracies, biases, or inappropriate expressions. Users are obligated to critically evaluate the output of LLMs and use it at their own risk. For critical decision-making, users should exercise caution and seek expert advice.

## 📄 License

This project is licensed under the [MIT License](https://github.com/yf591/llm-toolkit/blob/main/LICENSE).
However, pre-trained LLM models, datasets, and libraries used within each notebook may have their own respective licenses. Please ensure you review and comply with all applicable license terms.
