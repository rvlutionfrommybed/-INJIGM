# Professor Full Execution Guide

## 실행 순서

1. Colab에서 `professor_notebooks_01_to_06_combined.ipynb` 업로드
2. 런타임 유형을 GPU로 변경
3. Notebook 01부터 06까지 순서대로 실행
4. 너무 오래 걸리면 각 노트북의 `range(100)` 또는 `range(30)`만 줄여서 먼저 테스트
5. 제출용은 Notebook 06 TinyGPT의 loss 로그와 생성 샘플을 반드시 캡처

## 교수님 코드 흐름

- Notebook 01: Bigram Language Model on names.txt
- Notebook 02: MLP Character Model on names.txt
- Notebook 03: MLP on Tiny Shakespeare
- Notebook 04: GPT-style Dataset + Minimal Sequence Model
- Notebook 05: Single-Head Masked Self-Attention
- Notebook 06: Toward a Tiny GPT

## 보고서 핵심 문장

본 프로젝트는 교수님 노트북의 단계적 흐름을 따라 Bigram Language Model에서 시작하여 MLP Character Model, GPT-style sequence dataset, masked self-attention, multi-head attention 기반 TinyGPT까지 구현하였다.
