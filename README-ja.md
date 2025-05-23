#  LLM Toolkit 🛠️ - 日本語版

このリポジトリは、私が大規模言語モデル (LLM) の学習のために作成し実践したサンプルコード、および学習リソースをまとめたツールキットです。テキスト生成の基本から、ファインチューニング (Supervised Fine-Tuning: SFT)、RAG (Retrieval Augmented Generation) といった応用技術まで、LLMを深く活用するための様々なテクニックをGoogle Colabノートブック形式でまとめています。

LLM分野について、自分自身の知識およびスキルの向上を目指すとともに、この分野を学んでいる方が開発を始めるための一助となることを目指しています。初学者でも、ステップバイステップで学べるコンテンツになればと思い公開設定にしています。

## 🚀 提供ノートブック一覧

| ノートブック                                                                                                                               | 説明                                                                                                                                                           | 主要技術・ライブラリ                                                                                                             | 特徴・学べること                                                                                             | Colabで開く                                                                                                                                    |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Universal LLM GUI Notebook](https://github.com/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)**                      | 様々な事前学習済みLLMを選択し、パラメータを調整しながらテキスト生成を試せる、汎用的なGradio UI付きノートブック。                                                                    | `transformers`, `torch`, `gradio`                                                                                              | LLMの基本操作, モデル選択, パラメータ調整, 手軽なテキスト生成体験                                                                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb) |
| **[SFT LLM FineTuning GUI Notebook](https://github.com/yf591/llm-toolkit/blob/main/SFT_LLM_FineTuning_GUI_Notebook.ipynb)**               | `trl`ライブラリの`SFTTrainer`を使用し、QLoRAを用いて効率的にLLMを教師ありファインチューニングする手順を解説。Gradio UIでファインチューニング後のモデルを試せます。               | `transformers`, `peft` (QLoRA), `trl` (SFTTrainer), `bitsandbytes`, `datasets`, `gradio`, `torch`, `accelerate`                  | SFT (QLoRA), `SFTTrainer`活用, データセット整形, 効率的ファインチューニング, Gradio UIによる評価                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/SFT_LLM_FineTuning_GUI_Notebook.ipynb) |
| **[LLM SFT & RAG with GradioUI](https://github.com/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb)**                        | QLoRAを用いたSFTに加え、RAGを組み合わせて特定の専門知識に基づいて応答するチャットボットを構築。Gradio UIでインタラクティブに試せます。                                        | `transformers`, `peft` (QLoRA), `langchain`, `faiss-cpu`, `bitsandbytes`, `datasets`, `gradio`, `torch`, `accelerate`                | SFT (QLoRA) + RAG, 専門知識の統合, ベクトルDB (FAISS), LangChainパイプライン, 対話型UI, モデル共有                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/LLM_SFT_%26_RAG_GradioUI.ipynb) |
| *RLHF_Notebook.ipynb (仮)*                                                                                                              | *(準備中)* 人間のフィードバックからの強化学習 (RLHF: Reinforcement Learning from Human Feedback) を実践するためのノートブック。                                              | `trl` (予定), `transformers`, `datasets`                                                                                       | *(予定)* 報酬モデリング, PPOアルゴリズム, LLMの振る舞い調整                                                                       | *(Coming Soon)*                                                                                                                            |
| *DPO_Notebook.ipynb (仮)*                                                                                                               | *(準備中)* 直接的な選好最適化 (DPO: Direct Preference Optimization) を用いた、よりシンプルなアライメント手法を学ぶノートブック。                                                     | `trl` (予定), `transformers`, `datasets`                                                                                       | *(予定)* 選好ペアデータ学習, RLHFの代替                                                                                          | *(Coming Soon)*                                                                                                                            |

## 📚 学べること・試せる主要技術

このツールキットを通じて、以下のLLM関連技術の理解を深め、基本的なスキルを学ぶことができます。

*   **LLMの基礎と推論**: Hugging Face Hubから様々なモデルをロードし、テキスト生成を行う基本的なワークフロー。
*   **効率的ファインチューニング (Supervised Fine-Tuning: SFT)**
    *   **QLoRA (Quantized Low-Rank Adaptation)**: 低リソース環境でも大規模モデルを効率的にパラメータ調整する手法。
    *   **異なる実装アプローチの比較学習**
        *   **`trl` ライブラリの `SFTTrainer`** (`SFT_LLM_FineTuning_GUI_Notebook.ipynb` で使用)
            *   **強み**: SFT（特に対話・指示形式）の**標準的なユースケース**に対して、**手軽かつ効率的**に高い性能を出すためのツール。コードが簡潔になり、SFT特有のデータ処理（パッキング等）も容易。
            *   **考慮点**: 標準から逸脱した特殊なSFTや、訓練ループの非常に詳細なカスタマイズには限界がある場合も。
        *   **カスタムPyTorchトレーニングループ** (`LLM_SFT_&_RAG_GradioUI.ipynb` で使用)
            *   **強み**: **最大限の柔軟性と制御**。あらゆる種類のデータ形式（例: RAGで拡張したプロンプト）、訓練ロジック、損失関数などを自由に実装可能。特定の独自要件に合わせた高度なチューニングで、**潜在的な最高性能を追求できる**。
            *   **考慮点**: 実装の複雑性が増し、より多くの知識と試行錯誤が必要。`SFTTrainer` が提供するような便利な機能は自身で実装する必要がある。
    *   特定のタスクやカスタムデータセットへのモデル適応（指示チューニング、対話チューニング、RAG拡張チューニングなど）。
*   **RAG (Retrieval Augmented Generation)**
    *   外部ドキュメント（PDF、テキストファイル等）をLLMの知識源としてリアルタイムに活用する方法。
    *   **ベクトルデータベース (FAISS)**: ドキュメントをベクトル化し、高速な類似性検索を実現。
    *   **LangChain**: RAGパイプライン（ドキュメント読み込み、分割、埋め込み、検索、プロンプト生成、LLM連携）を効率的に構築・管理。
*   **インタラクティブUI開発**: Gradioを用いた、モデルを手軽に試せるWeb UIの作成。
*   **Hugging Face Hubとの連携**: 学習済みモデル（アダプタ）のHubへの共有方法。

## 🛠️ ノートブックの基本的な使い方

1.  **Colabでノートブックを開く**
    *   上記の表の「Colabで開く」列のバッジをクリックするか、各ノートブックのファイル名リンクから直接開いてください。
2.  **ランタイムの設定 (重要)**
    *   Colabメニューから「ランタイム」→「ランタイムのタイプを変更」。
    *   「ハードウェアアクセラレータ」で**「GPU」**（T4, L4, A100など、利用可能なもの）を選択してください。特にファインチューニングや大規模モデルの実行にはGPUが不可欠です。無料版ColabではT4 GPUが利用できる場合があります。
3.  **セットアップセルの実行**
    *   各ノートブックの冒頭には、必要なライブラリのインストールや初期設定を行うセルがあります。これらを順番に実行してください。
    *   **注意**: SFTやRAGを含むノートブックは多くのライブラリをインストールするため、初回のセットアップには数分かかることがあります。
4.  **データの準備 (ノートブックによる)**
    *   `SFT_LLM_FineTuning_GUI_Notebook.ipynb` や `LLM_SFT_&_RAG_GradioUI.ipynb` など、ファインチューニングを行うノートブックでは、学習用のデータセットやRAG用のドキュメントファイルをColab環境にアップロードしたり、特定のパスに配置したりする必要があります。ノートブック内の指示に従い、必要なデータを準備してください。
5.  **セルを順番に実行**
    *   各セルを上から順に実行し、Markdownセルに記載された説明や指示を読みながら進めてください。
    *   Gradio UIが起動するセルを実行すると、Colabの出力エリアにWeb UIが表示され、モデルを試すことができます。

## 📖 各ノートブックのハイライト

### 1. Universal LLM GUI Notebook 💬
様々な事前学習済みLLM（例: `elyza/ELYZA-japanese-Llama-2-7b-instruct`, `google/gemma-7b-it` 等）を簡単にロードし、テキスト生成を試すための汎用ノートブック。プロンプト入力や生成パラメータ（温度、最大トークン数など）を調整できるGradio UIが付属します。

**こんな人におすすめ**:
*   色々なLLMを手軽に試してみたい。
*   プロンプトエンジニアリングの練習をしたい。
*   Gradioでの簡単なUI構築を学びたい。

*(ノートブック内のコード例は、汎用性を高めるため、特定のモデルに依存しない形で記述されています。)*

### 2. SFT LLM FineTuning GUI Notebook 🎯
このノートブックでは、Hugging Faceの `trl` ライブラリに含まれる `SFTTrainer` を活用し、QLoRA (Quantized Low-Rank Adaptation) を用いて効率的に大規模言語モデル（例: `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1`）を教師ありファインチューニング (SFT) する具体的な手順をステップバイステップで解説します。カスタムデータセットの準備から、学習設定、トレーニング実行、そしてファインチューニング後のモデルの動作確認までを網羅し、最後にGradio UIで手軽に推論を試すことができます。
`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1`ではHugging Faceへのログインのセルの実行は不要ですが、モデルによってはHugging Faceへのログインが必要な場合があります。

**主なステップと学習ポイント**
1.  **環境準備**: GPU確認、ライブラリインストール、Hugging Face Hubログイン。
2.  **モデルとトークナイザーの準備**: ベースモデルのロードと4ビット量子化 (QLoRA用)。
3.  **データセットの準備**: カスタムデータセット（指示と応答のペア）をロードし、`SFTTrainer` が要求するチャット形式に整形。
4.  **QLoRAと学習設定**: LoRAのターゲットモジュール特定、`LoraConfig` および `TrainingArguments` の設定。
5.  **`SFTTrainer`によるファインチューニング**: 数行のコードで効率的なSFTを実行。
6.  **結果確認とモデル保存**: 学習済みLoRAアダプタの保存と、トレーニング可能なパラメータ数の確認。
7.  **ファインチューニング済みモデルでの推論**: 保存したアダプタをベースモデルに適用し、Gradio UIで対話的に性能を評価。

**こんな人におすすめ**
*   特定の指示応答データセットでLLMをファインチューニングしたい。
*   QLoRAと**`SFTTrainer`を使った手軽で効率的な**学習方法を習得したい。
*   日本語LLMのカスタマイズに挑戦したい。
*   限られた計算リソースでSFTを試したい。

### 3. LLM SFT & RAG with GradioUI 🧠📚
このノートブックは、上記SFTの技術をさらに発展させ、RAG (Retrieval Augmented Generation) と組み合わせることで、LLMに外部の専門知識を動的に組み込む方法を探求します。日本語LLM（例: `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1`）をベースに、特定のドキュメント（例: "Reinforcement Learning: An Introduction"のPDF）を知識源として活用するチャットボットを構築しています。
※日本語LLMに英語のRAGを実装した場合にどのような精度になるのかという実験的な意味合いも含まれています。

**主なステップと学習ポイント**
1.  **RAGシステム構築**: あなたが用意した専門ドキュメントを知識源とし、FAISSを用いたベクトルデータベースを構築。ドキュメントローディング、チャンキング、エンベディング、インデックス作成までをカバー。
2.  **SFT用データ準備**: 指示応答データセット（例: `ichikara-instruction`）の各質問に対し、上記RAGシステムで検索した関連コンテキストを付加。これにより、モデルは「文脈を読んで答える」訓練を受けます。
3.  **QLoRAによるSFT**: ベースモデルに対し、準備したRAG拡張データセットを用いて、メモリ効率の良いファインチューニング（QLoRA）を実行。このノートブックでは、カスタムPyTorchトレーニングループ（またはHugging Face Trainer）を使用して、RAGに特化したデータ形式と訓練プロセスを柔軟に制御する手法を学びます。
4.  **推論パイプライン構築**: ファインチューニングされたLoRAアダプタとRAGシステムをLangChainで統合し、質問応答パイプラインを構築。
5.  **Gradio UIでの対話型デモ**: 作成したRAGチャットボットとGradioを通じて対話し、その性能を定性的に評価。
6.  **Hugging Face Hubへのアダプタ共有**: 学習したLoRAアダプタをHubにアップロードし、成果を共有する手順。

**こんな人におすすめ**
*   特定の専門知識を持つLLMを構築し、その知識に基づいて応答させたい。
*   SFTとRAGを組み合わせた高度なLLMカスタマイズ手法を学び、**訓練ループのカスタマイズにも挑戦したい**。
*   LangChainとベクトルデータベースを用いたRAGの実装を理解したい。
*   実践的なLLMアプリケーション開発のレシピを知りたい。

## 💡 今後のロードマップ

*   **RLHF (Reinforcement Learning from Human Feedback)** ノートブックの追加: より自然でユーザーの意図に沿った応答を生成するための強化学習。
*   **DPO (Direct Preference Optimization)** ノートブックの追加: RLHFの複雑さを軽減し、選好データから直接LLMをファインチューニングする手法。
*   様々なサイズのモデル（軽量モデルから大規模モデルまで）への対応。
*   多様なタスク（要約、翻訳、コード生成など）に対応したサンプルノートブックの拡充。
*   各ノートブックの解説ドキュメント（理論的背景、実装の詳細など）の充実。
*   モデル評価のための具体的なメトリクスや手法の導入。

## 🙏 コントリビューションについて

バグ報告、機能改善の提案、新しいサンプルノートブックの追加、ドキュメントの修正など、いろんな形でのコントリビューションをお待ちしています！お気軽にGitHubのIssueを作成したり、Pull Requestを送ってください。このツールキットをより良いものにしていきたいと思っています。

## 📜 免責事項

本リポジトリで提供されるソフトウェアおよびサンプルコードは「現状のまま」で提供され、商品性、特定の目的への適合性、および非侵害を含むがこれらに限定されない、明示的または黙示的な一切の保証を行いません。作者または著作権者は、本ソフトウェアの使用またはその他の取引に起因または関連して発生した、いかなる請求、損害、またはその他の責任についても一切責任を負わないものとします。
大規模言語モデル (LLM) の生成する内容には、不正確な情報、偏見、あるいは不適切な表現が含まれる可能性があります。利用者は、LLMの出力を常に批判的に評価し、自己の責任において利用する義務があります。特に重要な意思決定に際しては、専門家の助言を求めるなど、慎重な判断を心がけてください。

## 📄 ライセンス

このプロジェクトは [MIT License](https://github.com/yf591/llm-toolkit/blob/main/LICENSE) の下でライセンスされています。
ただし、各ノートブック内で使用される事前学習済みLLMモデル、データセット、およびライブラリには、それぞれ独自のライセンスが付与されている場合があります。それらのライセンス条項を十分に確認し、遵守してください。
