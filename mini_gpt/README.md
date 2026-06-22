# Mini GPT Project

수업에서 배운 PyTorch 학습 흐름을 바탕으로 만든 character-level mini GPT입니다.

## 핵심 아이디어

- 텍스트를 글자 단위 토큰으로 변환합니다.
- 모델은 이전 글자들을 보고 다음 글자를 예측합니다.
- causal self-attention을 사용해서 미래 토큰을 보지 못하게 합니다.
- `loss.backward()`와 `AdamW`로 학습합니다.

## 실행 방법

Colab 또는 torch가 설치된 환경에서 실행합니다.

```bash
pip install -r requirements.txt
python train.py
python generate.py
```

학습 결과는 다음 파일에 저장됩니다.

- `logs/train_log.txt`
- `logs/sample_output.txt`
- `mini_gpt.pt`

## 수업 내용과의 연결

Week 3~4에서 다룬 `torch`, `nn.Module`, `Linear`, `DataLoader`식 학습 흐름, `loss.backward()`, optimizer update를 Transformer 기반 언어모델에 적용한 확장 실습입니다.
