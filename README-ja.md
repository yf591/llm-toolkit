# LLM Toolkit (日本語版)

このリポジトリは、大規模言語モデル (LLM) のための私が実践したツール、サンプルコード、および学習リソースをまとめたツールキットです。テキスト生成、ファインチューニング、RAG (Retrieval Augmented Generation) など、LLMを活用するための様々なテクニックをGoogle Colabノートブック形式にまとめています。

初心者でも、LLMの可能性を探求して独自の応用を開発するための出発点となることを目指しています。

## 🚀 提供するノートブック

| ノートブック                                                                                                                               | 説明                                                                                                                                                           | 主要技術・ライブラリ                                                                                                             | 特徴・学べること                                                                                             | Colabで開く                                                                                                                                    |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Universal LLM GUI Notebook](https://github.com/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)**                      | 様々な事前学習済みLLMを選択し、パラメータを調整しながらテキスト生成を試せる、汎用的なGradio UI付きノートブック。                                                                    | `transformers`, `torch`, `gradio`                                                                                              | LLMの基本操作, モデル選択, パラメータ調整, 手軽なテキスト生成体験                                                                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb) |
| **[LLM SFT & RAG with GradioUI](https://github.com/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb)**                        | QLoRAを用いた効率的なSFTと、RAGを組み合わせ、特定の専門知識に基づいて応答するチャットボットを構築。Gradio UIでインタラクティブに試せます。                                  | `transformers`, `peft` (QLoRA), `langchain`, `faiss-cpu`, `bitsandbytes`, `datasets`, `gradio`, `torch`, `accelerate`                | SFT (QLoRA), RAG実装, 専門知識の統合, ベクトルDB (FAISS), LangChainパイプライン, 対話型UI, モデル共有                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb) |
| *RLHF_Notebook.ipynb (仮)*                                                                                                              | *(準備中)* 人間のフィードバックからの強化学習 (RLHF: Reinforcement Learning from Human Feedback) を実践するためのノートブック。                                              | `trl` (予定), `transformers`, `datasets`                                                                                       | *(予定)* 報酬モデリング, PPOアルゴリズム, LLMの振る舞い調整                                                                       | *(Coming Soon)*                                                                                                                            |
| *DPO_Notebook.ipynb (仮)*                                                                                                               | *(準備中)* 直接的な選好最適化 (DPO: Direct Preference Optimization) を用いた、よりシンプルなアライメント手法を学ぶノートブック。                                                     | `trl` (予定), `transformers`, `datasets`                                                                                       | *(予定)* 選好ペアデータ学習, RLHFの代替                                                                                          | *(Coming Soon)*    

## 📚 学習できること・試せること

このツールキットを通じて、以下のようなLLM関連技術の基礎を学ぶことができます。

*   **基本的なLLMの利用**: Hugging Face Hubからモデルをロードし、テキストを生成する基本的な流れ。
*   **ファインチューニング (Supervised Fine-Tuning)**
    *   QLoRA (Quantized Low-Rank Adaptation) を用いた効率的なパラメータチューニング。
    *   特定のタスクやデータセットへのモデルの適応。
*   **RAG (Retrieval Augmented Generation)**
    *   外部ドキュメント（PDFなど）を知識源としてLLMに組み込む方法。
    *   ベクトルデータベース (FAISS) の構築と利用。
    *   LangChainを用いたRAGパイプラインの実装。
*   **インタラクティブなUIの構築**: Gradioを用いた簡単なWeb UIの作成。
*   **モデルの評価とデバッグ**: (各ノートブック内で部分的に触れています)
*   **Hugging Face Hubへのモデル共有**: (LLM_SFT_&_RAG_GradioUI.ipynb の最終ステップで解説)

## 🛠️ 使い方 (各ノートブック共通)

1.  **Colabでノートブックを開く**
    *   上記の表から試したいノートブックのリンクをクリックするか、以下のバッジをご利用ください。
        *  [テキスト生成](https://github.com/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)
          
        *  [ファインチューニング（QLoRA）](https://github.com/yf591/llm-toolkit/blob/main/SFT_LLM_FineTuning_GUI_Notebook.ipynb)
        
        *  [ファインチューニング＆RAG](https://github.com/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb)
2.  **ランタイムの設定**
    *   Colabメニューから「ランタイム」→「ランタイムのタイプを変更」を選択。
    *   「ハードウェアアクセラレータ」で「GPU」（T4, L4, A100など）を選択してください。特にファインチューニングや大規模モデルの実行にはGPUが必須です。
3.  **セットアップセルの実行**
    *   各ノートブックの最初の数セルには、必要なライブラリのインストールや環境設定のコードが含まれています。これらを順番に実行してください。
    *   **注意:** `LLM_SFT_&_RAG_GradioUI.ipynb` は多くのライブラリをインストールするため、初回実行には時間がかかる場合があります。
4.  **データの準備 (必要な場合)**
    *   `LLM_SFT_&_RAG_GradioUI.ipynb` では、ファインチューニング用のデータセットやRAG用のドキュメントファイルをColab環境にアップロードする必要があります。ノートブック内の指示に従ってください。
5.  **ノートブックの実行**
    *   各セルを上から順に実行していきます。Markdownセルに説明が記載されているので、それを読みながら進めてください。
    *   Gradio UIが起動するセルを実行すると、Colabの出力エリアにインタラクティブなUIが表示されます。

## 📖 各ノートブックの詳細

### 1. Universal LLM GUI Notebook

様々な事前学習済みLLM（例: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`）を簡単に試し、テキスト生成を行うための汎用的なノートブックです。GradioベースのシンプルなGUIが提供され、プロンプト入力やパラメータ調整が可能です。

**主な機能**
*   Hugging Face Hubからのモデルとトークナイザーのロード。
*   テキスト生成関数の実装。
*   Gradioを用いたインタラクティブなUI。

**使用例 (抜粋)**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# モデルとトークナイザーのロード (ノートブック内でモデル名は選択可能)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # or torch.bfloat16
    device_map="auto"
)

def generate_text(prompt, max_new_tokens=200): # max_input_length はトークナイザ側で制御
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio インターフェース (ノートブック内の実装はより多機能です)
# interface = gr.Interface(...)
# interface.launch(share=True)
```

### 2. LLM SFT & RAG with Gradio UI

このノートブックは、より高度なLLMのカスタマイズと応用を探求します。
`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1` などの日本語LLMをベースに、以下のステップを実行します。

1.  **RAGシステムの構築**: 特定のドキュメント（例: 専門書PDF）を知識源として、FAISSを用いたベクトルデータベースを構築します。
2.  **SFT用データの準備**: 指示データセット（例: `ichikara-instruction`）の各項目に対し、RAGで検索したコンテキストを付加して学習データを作成します。
3.  **QLoRAによるSFT**: ベースモデルに対し、準備したRAG拡張データセットを用いて効率的なファインチューニング（QLoRA）を行います。
4.  **推論パイプラインの構築**: ファインチューニングされたLoRAアダプタとRAGシステムを組み合わせた推論パイプラインをLangChainで構築します。
5.  **Gradio UIでの対話**: 作成したRAGチャットボットとGradioを通じて対話し、性能を確認できます。
6.  **Hugging Face Hubへのアップロード**: 学習したLoRAアダプタをHubに共有する手順も含まれます。

**主な学習ポイント**
*   LoRA/QLoRAによるメモリ効率の良いファインチューニング。
*   LangChainを用いたRAGパイプラインの設計と実装。
*   FAISSベクトルストアの作成と利用。
*   特定の知識ドメインに特化したLLMの構築。
*   インタラクティブなデモアプリケーションの作成。

このノートブックは、LLMを特定の専門知識に対応させるための実践的なレシピを提供します。

## 💡 今後の予定

*   RLHF (Reinforcement Learning from Human Feedback) のノートブック追加。
*   DPO (Direct Preference Optimization) のノートブック追加。
*   より多様なモデルやタスクに対応したサンプルの拡充。
*   各ノートブックの解説ドキュメントの充実。

## 🙏 コントリビューション

バグ報告、機能提案、新しいノートブックの追加など、あらゆるコントリビューションを歓迎します！IssueやPull Requestでお気軽にご参加ください。

## 📜 免責事項

本ソフトウェアは「現状のまま」で提供され、明示または黙示のいかなる種類の保証もありません。商品性、特定の目的への適合性、および非侵害を含むがこれらに限定されません。作者または著作権者は、本ソフトウェアまたは本ソフトウェアの使用もしくはその他の取引に起因または関連して発生した、いかなる請求、損害、またはその他の責任についても責任を負わないものとします。LLMの生成する内容には不正確な情報や偏見が含まれる可能性があるため、利用者はその出力を批判的に評価する責任があります。

## 📄 ライセンス

このプロジェクトは [MIT License](https://github.com/yf591/llm-toolkit/blob/main/LICENSE) の下でライセンスされています。ただし、使用する各LLMモデルやデータセットにはそれぞれ独自のライセンスが付与されている場合があるため、それらのライセンス条件も遵守してください。
