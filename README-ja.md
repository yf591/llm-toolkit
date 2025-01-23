# llm-toolkit (日本語)

大規模言語モデル（LLM）を扱うためのツールとレシピ集です。このリポジトリは、以下のようなタスクのための様々なユーティリティと例を提供します。

*   テキスト生成
*   教師ありファインチューニング（SFT）（※追加時期は未定）
*   人間からのフィードバックを用いた強化学習（RLHF）（※追加時期は未定）
*   Direct Preference Optimization（DPO）（※追加時期は未定）

## はじめに

このツールキットを使い始めるには、以下の手順に従ってください。

1.  **Colab で開く:** 下の「Open in Colab」ボタンをクリックして、Google Colab でノートブックを開きます。

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yf591/llm-toolkit/blob/main/Universal_LLM_GUI_Notebook.ipynb)

2.  **ランタイムタイプの設定:** Colab で、「ランタイム」->「ランタイムのタイプを変更」を選択し、ハードウェアアクセラレータとして「GPU」を選択します。特に大規模モデルを使用する場合は、パフォーマンス向上のために強く推奨されます。

3.  **ライブラリのインストール:** Colab のコードセルで以下のコマンドを実行し、必要なライブラリをインストールします。

    ```bash
    !pip install transformers torch gradio
    ```

4.  **Google ドライブのマウント (オプション):** Google ドライブ内のファイルにアクセスする必要がある場合は、Colab のコードセルで以下のコードを実行し、認証プロンプトに従ってください。

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

5.  **ノートブックの実行:** ノートブック内のコードセルを実行します。

## 例

### テキスト生成

以下のコードは、事前学習済み言語モデルを使用した基本的なテキスト生成を示しています。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# モデルとトークナイザーのロード
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

# Gradio インターフェース
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="プロンプト", lines=10, max_lines=20),
        gr.Slider(minimum=128, maximum=2048, value=1024, label="最大入力長"),
        gr.Slider(minimum=50, maximum=500, value=200, label="最大出力長")
    ],
    outputs=gr.Textbox(label="生成されたテキスト", lines=10, max_lines=20)
)

interface.launch(share=True)
```
## 免責事項

本ソフトウェアは「現状のまま」で提供され、商品性、特定の目的への適合性、および非侵害を含むがこれらに限定されない、明示または黙示のいかなる種類の保証もありません。作者または著作権者は、契約、不法行為、またはその他の行為であるかどうかにかかわらず、ソフトウェアまたはソフトウェアの使用もしくはその他の取引に起因または関連して発生した、いかなる請求、損害、またはその他の責任についても責任を負わないものとします。

## ライセンス

このプロジェクトは MIT License の下でライセンスされています - 詳細は [LICENSE](https://github.com/yf591/llm-toolkit/blob/main/LICENSE) ファイルをご覧ください。
