# Today Execution Checklist

## Goal

교수님 노트북 흐름에 맞춘 TinyGPT를 실행하고, 제출 가능한 증거를 확보한다.

## 1. 먼저 읽을 것

- `professor_notebook_summary.md`
- `tiny_gpt_professor_style.py`

## 2. Colab 또는 torch 설치 환경에서 실행

빠른 테스트:

```bash
python tiny_gpt_professor_style.py --quick
```

제출용 실행:

```bash
python tiny_gpt_professor_style.py --epochs 20 --max-steps 300
```

시간이 부족하면:

```bash
python tiny_gpt_professor_style.py --epochs 10 --max-steps 200
```

## 3. 캡처할 것

- `xb.shape`, `yb.shape`
- `logits.shape`
- epoch별 train loss
- 생성된 `ROMEO:` 샘플 텍스트

## 4. 저장되는 파일

- `logs/professor_style_train_log.txt`
- `logs/professor_style_sample.txt`
- `professor_style_tiny_gpt.pt`

## 5. 보고서에 반드시 넣을 말

> 본 프로젝트는 수업 노트북의 흐름에 따라 Bigram Language Model에서 시작해 MLP Character Model, GPT-style sequence dataset, single-head masked self-attention을 거쳐 multi-head attention 기반 TinyGPT를 구현하였다.

> 최종 TinyGPT는 token embedding, positional embedding, masked multi-head self-attention, feedforward network, residual connection, layer normalization, language modeling head로 구성된다.

## 6. 한 줄 설명

TinyGPT는 이전 토큰들을 보고 다음 토큰을 예측하는 autoregressive character-level language model이다.
