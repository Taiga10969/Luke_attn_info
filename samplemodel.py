TOKENIZER_NAME = "studio-ousia/luke-base"
MODEL_NAME = "studio-ousia/luke-base"
import transformers
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List

from transformers import AutoTokenizer
from LukeClassifier import LukeForSequenceClassification
from torchinfo import summary
import logging

logging.disable(logging.INFO)
logging.disable(logging.WARNING)


def main():
    # 引数としてはBERTの時と同じでnum_labelsで指定する
    model = LukeForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # *save
    # model.save_pretrained("./model")
    # tokenizer.save_pretrained("./model")

    # *input(3)
    sample_ds = "今回はフーリエ変換を学んだ．", "よくわかった", "わからなかった"

    # *tokenize
    inputs = tokenizer(
        sample_ds,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    print("inputs['input_ids'][0] : ", inputs["input_ids"][0])

    # *モデルのsummaryを表示
    #summary(model, depth=4)

    # *モデルのパラメータを確認
    #print(f"モデルのパラメータ数: {model.num_parameters()}")

    # *モデル出力
    inputs.to("cuda")
    model.to("cuda")
    print(f"start model output")

    attn_info = torch.randn(3,12,12,24,24).to('cuda')


    #print('**inputs : ', **inputs)
    with torch.no_grad():
        outputs = model(**inputs, attn_info=attn_info)

    # *モデルの出力を確認
    print(f"outputs : {outputs}", type(outputs))
    print(f"outputs.logits : {outputs.logits}", type(outputs.logits))

    # * 結果の取得
    predictions = outputs.logits
    predicted_class = torch.argmax(predictions, dim=1)
    print(f"predicted_class: {predicted_class}")


if __name__ == "__main__":
    main()
