"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"

    uv run python -m app.ask "ポケットモンスターについて教えて"　
        [トークン数] 入力: 8, 出力: 655 | [応答時間] 3.24秒
    uv run python -m app.ask "現在のポケモンカードゲームの27年シーズンの環境における、ドラパルトデッキの主な構成と特徴を教えて‌"
        [トークン数] 入力: 26, 出力: 1418 | [応答時間] 7.06秒
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    # 質問の後は、空欄を入力するまで会話履歴を保って続けて質問できる

ツールは使わず、同じチャットで会話履歴を保ちながら質問する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: Chat) -> str:
    start_time = time.perf_counter()
    response = chat.send_message(question)
    elapsed_time = time.perf_counter() - start_time

    usage = response.usage_metadata
    if usage is None:
        input_tokens = "未取得"
        output_tokens = "未取得"
    else:
        input_tokens = (
            usage.prompt_token_count
            if usage.prompt_token_count is not None
            else "未取得"
        )
        output_tokens = (
            usage.candidates_token_count
            if usage.candidates_token_count is not None
            else "未取得"
        )
    print(
        f"[トークン数] 入力: {input_tokens}, 出力: {output_tokens}"
        f" | [応答時間] {elapsed_time:.2f}秒"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, chat))
            try:
                question = input("\n次の質問（終了するには空欄）: ").strip()
            except EOFError:
                break
            if not question:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
【呼び出せるモデル全件】
### 1. 主力・最新モデル（Gemini 1.5シリーズ）
現在、最も広く使われており、膨大なデータを一度に処理できる（長文コンテキスト）のが特徴の最新世代です。

* **Gemini 1.5 Pro**
  * **特徴:** 中規模〜高難度のタスク向けの中核モデル。非常に長い文章やコード（最大100万〜200万トークン以上）を一度に理解でき、複雑な推論やプログラミング、クリエイティブな文章作成が得意です。
* **Gemini 1.5 Flash**
  * **特徴:** 軽量・高速・低コストに特化したモデル。1.5 Proの長文処理能力を引き継ぎつつ、スピードと効率を重視しています。チャットボットのようにリアルタイムの返答が求められる場面でよく使われます。

### 2. 軽量・エッジデバイス向けモデル（Gemmaシリーズなど）
サーバーだけでなく、スマホやパソコンなどのローカル環境でも動かせるように作られたオープンなモデルです。

* **Gemma 2 (2B / 9B / 27B)**
  * **特徴:** Googleがオープンソースとして公開している軽量モデル（パラメータ数ごとに20億、90億、270億などのサイズがある）。研究やローカルでの開発によく使われます。
* **CodeGemma**
  * **特徴:** プログラミング（コード生成・補完）に特化したGemma派生モデル。

### 3. 旧世代・特定の用途向けモデル（Gemini 1.0シリーズなど）
初期のモデルや、特定の実験的なバージョンです。

* **Gemini 1.0 Pro**
  * **特徴:** 最初に広く一般公開されたProモデル。現在は主に1.5シリーズへ移行が進んでいます。
* **Gemini 1.0 Ultra**
  * **特徴:** 最も巨大で最高峰の性能を持つとされたモデルですが、現在は1.5 Proやさらに新しいモデルに役割が引き継がれつつあります。

---

### 💡 補足：APIや開発者向けとしての「全件」
もしあなたがGoogle AI Studio（開発者向けツール）やVertex AI（Google Cloud）などのAPI経由で呼び出す場合、これらに加えて以下のような細かいバージョン違い（エイリアス）が存在します。

* `gemini-1.5-pro-latest` （常に最新の1.5 Proを指す）
* `gemini-1.5-flash-latest` （常に最新の1.5 Flashを指す）
* `gemini-pro` （1.0 Pro）
* `gemini-pro-vision` （画像認識に特化した1.0世代のモデル）
"""