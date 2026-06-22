# Professor Notebook Summary

이 파일은 `notebook_01.py`부터 `notebook_06.py`까지의 흐름을 기말 프로젝트 기준으로 정리한 것이다.

## 결론

교수님이 요구하는 GPT 구현은 full-scale GPT-2 pretraining이 아니라, 수업 노트북의 단계적 흐름을 따라가는 **Tiny GPT 직접 구현**이다.

핵심은 다음 세 가지다.

1. 문자를 숫자로 바꾸고 next-token prediction 문제를 만든다.
2. PyTorch `nn.Module`, `Dataset`, `DataLoader`, `loss.backward()`, `AdamW`로 학습한다.
3. 마지막에는 masked self-attention, multi-head attention, feedforward, residual connection, layer normalization을 포함한 Tiny GPT를 구현한다.

## Notebook 1: Bigram Language Model

- 데이터: `names.txt`
- 입력: 현재 문자 1개
- 출력: 다음 문자 1개
- 모델: bigram 확률표
- 구현 포인트:
  - `Dataset`
  - `DataLoader`
  - one-hot 또는 `nn.Embedding`
  - cross entropy loss
  - sampling

## Notebook 2: MLP Character Model

- 데이터: `names.txt`
- 입력: 길이 `block_size`의 context
- 출력: 다음 문자 1개
- 모델: `Embedding -> Flatten -> Linear -> Tanh -> Linear`
- 의미:
  - bigram보다 긴 문맥을 사용한다.
  - GPT 전 단계인 context 기반 language model이다.

## Notebook 3: MLP on Tiny Shakespeare

- 데이터: `tiny Shakespeare`
- 모델은 Notebook 2의 MLP를 유지한다.
- 바뀐 점:
  - 이름 데이터에서 긴 텍스트 데이터로 이동한다.
  - sliding-window dataset을 사용한다.
- 한계:
  - MLP는 fixed context만 보므로 긴 문맥 처리에 약하다.

## Notebook 4: GPT-style Dataset + Minimal Sequence Model

- 가장 중요한 전환점이다.
- 기존:
  - `x = context`
  - `y = next char`
- GPT-style:
  - `x = [t1, t2, ..., tT]`
  - `y = [t2, t3, ..., t(T+1)]`
- 출력 shape:
  - `(B, T, V)`
- 새로 등장:
  - token embedding
  - positional embedding
  - sequence cross entropy
- 아직 attention은 없다.

## Notebook 5: Single-Head Masked Self-Attention

- GPT의 핵심인 masked self-attention을 추가한다.
- 구현 포인트:
  - `key`, `query`, `value`
  - attention score: `q @ k.transpose(-2, -1)`
  - scaling
  - causal mask: `torch.tril`
  - softmax
  - weighted sum: `wei @ v`
- 의미:
  - 각 위치가 이전 위치들을 참고한다.
  - 미래 토큰은 mask로 차단한다.

## Notebook 6: Toward a Tiny GPT

- 최종 Tiny GPT 구조다.
- 추가되는 모듈:
  - multi-head attention
  - feedforward network
  - residual connection
  - layer normalization
  - block stacking
- 최종 모델:
  - token embedding
  - position embedding
  - stacked Transformer blocks
  - final layer norm
  - language modeling head
- 이 단계가 기말 과제의 핵심 구현 목표다.

## 제출 방향

보고서에서는 다음과 같이 설명하는 것이 가장 안전하다.

> 본 프로젝트는 GPT-2의 전체 규모를 재현하는 것이 아니라, 수업 노트북의 단계적 흐름을 따라 bigram language model에서 시작하여 MLP character model, GPT-style sequence dataset, masked self-attention, multi-head attention을 거쳐 Tiny GPT를 직접 구현하는 것을 목표로 한다.

## 평가자가 볼 가능성이 높은 체크포인트

- `Dataset`과 `DataLoader`를 사용하는가
- `x`, `y`가 GPT-style next-token sequence로 구성되는가
- output shape가 `(B, T, V)`인가
- sequence cross entropy를 올바르게 쓰는가
- causal mask로 미래 토큰을 차단하는가
- `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `TinyGPT` 구조가 있는가
- 학습 loss가 감소하는 로그가 있는가
- `sample_gpt` 또는 `generate`로 텍스트를 생성하는가

