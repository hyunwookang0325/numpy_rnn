# numpy_rnn — Character-level RNN / GRU in NumPy

문자 단위 언어 모델(이름 생성)을 **NumPy로 forward / backward(BPTT) / SGD / checkpoint까지 직접 구현**한 패키지입니다.
같은 모델을 PyTorch(`nn.RNN` / `nn.GRU` / `nn.LSTM`)로도 만들어 두고, 그것을 정답지로 삼아 NumPy 구현의 기울기를 검증합니다.

이전 저장소 [GRU_numpy](https://github.com/hyunwookang0325/GRU_numpy)(GRU 단일 파일 버전)를 패키지 구조로 옮기고, vanilla RNN을 추가했습니다.

```text
"emma", "olivia", ...  →  tokenizer  →  RNN / GRU (NumPy)  →  fc (H → 29)  →  softmax  →  다음 글자
                                          └ backward(BPTT)는 직접 유도 (numpy_rnns.py 주석 참고)
```

## 상태

| 모델 | NumPy forward / backward | torch 기울기 비교 | NumPy 학습 + npz 저장 | torch 버전 |
|---|---|---|---|---|
| RNN | ✅ | ✅ | ✅ | ✅ |
| GRU | ✅ | ✅ 15 / 15 | ✅ | ✅ |
| LSTM | ⬜ (예정) | ⬜ | ⬜ | ✅ |

## 구조

```text
numpy_rnn/
├── pyproject.toml            # src-layout 패키지 + pixi 환경
├── data/names.txt            # 이름 32,033개
└── src/numpy_rnn/
    ├── numpy_rnns.py         # NumPy RNN, GRU (forward, backward, step, train)
    ├── torch_rnns.py         # PyTorch RNN / GRU / LSTM + train() (정답지)
    └── utils.py              # tokenizer, train/val 분할, DataLoader, softmax, CrossEntropyLoss
```

## 설치 / 실행

[pixi](https://pixi.sh) 사용:

```bash
pixi install
pixi run python -m numpy_rnn.numpy_rnns     # NumPy 학습 (main()의 name = "RNN" | "GRU")
pixi run torch-rnn                          # PyTorch 학습
```

conda 사용:

```bash
pip install -e .
python -m numpy_rnn.numpy_rnns
```

패키지 안에서 상대 import(`from .utils import ...`)를 쓰므로, 파일을 직접 실행(`python numpy_rnns.py`)하지 말고 `python -m`으로 실행해야 합니다.

## 데이터

| id | 토큰 |
|---|---|
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3 – 28 | a – z |
| −1 | IGNORE (loss에서 제외) |

```text
"emma"  →  입력 X : [BOS, e, m, m, a, PAD, ...]   one-hot, [N, T, 29]   (T = 가장 긴 이름 + 1 = 16)
           정답 y : [e,   m, m, a, EOS, −1, ...]  class index, [N, T]
```

train / val = 8 : 2 (`random_state=42`), batch 16. 배치 처리에만 torch `DataLoader`를 쓰고, 계산은 전부 NumPy로 합니다.

## 모델

RNN:

```text
h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)
```

GRU (`torch.nn.GRU`와 같은 게이트 순서 r, z, n):

```text
r_t = σ(x_t W_xr^T + b_xr + h_{t-1} W_hr^T + b_hr)
z_t = σ(x_t W_xz^T + b_xz + h_{t-1} W_hz^T + b_hz)
n_t = tanh(x_t W_xn^T + b_xn + r_t ⊙ (h_{t-1} W_hn^T + b_hn))
h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

공통:

```text
X [B, T, 29] → RNN / GRU → h [B, T, H] → fc → logits [B·T, 29] → softmax → CrossEntropy (IGNORE −1)
```

- 가중치 초기값은 torch 레이어의 `state_dict()`에서 꺼내 씁니다. 그래서 같은 가중치로 torch autograd와 1:1로 비교할 수 있습니다.
- 한 모델 = 한 객체: fc 가중치도 class 안에 있고, `train()`이 fc 계산과 업데이트까지 합니다. `forward` / `backward` / `step`은 순환층만 담당합니다.

### Backward (BPTT)

```text
dh    = doutput[:, t] + dh_next
RNN : da = dh ⊙ (1 − h_t²)
GRU : dz = dh ⊙ (h_prev − n_t)          dan = dh ⊙ (1 − z_t) ⊙ (1 − n_t²)
      dr = dan ⊙ thn_t                  dthn = dan ⊙ r_t
      dar = dr ⊙ r_t(1 − r_t)           daz = dz ⊙ z_t(1 − z_t)
dW   += da^T @ 입력          (y = X W^T + b  →  dW = dy^T X, 시점마다 누적)
db   += Σ_batch da
GRU dh_prev = dar W_hr + daz W_hz + dthn W_hn + dh ⊙ z_t     (4갈래)
```

`CrossEntropyLoss`는 softmax와 합친 기울기 `(p − one_hot) / n_valid`를 돌려주고, padding(−1) 행은 0으로 만듭니다.

### Checkpoint

validation loss가 최저일 때 `{output_name}.npz`로 저장합니다 (기본 `numpy_rnn.npz`, `numpy_gru.npz`).

| 모델 | 키 |
|---|---|
| RNN | `weight_ih, weight_hh, bias_ih, bias_hh, weight_fc, bias_fc` |
| GRU | `Wxr, Wxz, Wxn, Whr, Whz, Whn, bxr, bxz, bxn, bhr, bhz, bhn, weight_fc, bias_fc` |
| 공통 | `epoch, val_loss, lr, hidden_size` |

```python
ckpt = np.load("numpy_gru.npz")
ckpt["Wxr"], float(ckpt["val_loss"])
```

## 검증

| 항목 | 결과 |
|---|---|
| GRU 기울기 vs `torch.autograd` | **15 / 15 일치** (GRU 12개 + dx + fc 2개), B = 5, T = 7, H = 11, padding 포함, max abs diff ≤ 7.5e-9 |
| RNN 기울기 vs `torch.autograd` | 일치, max abs diff ≤ 1.9e-6 |
| 학습 전 첫 배치 loss | 3.35 ~ 3.41 ≈ ln 29 = 3.37 (균등 추측) |

B, T, H를 일부러 서로 다르게 잡았습니다. B = H처럼 정사각형이면 전치를 잘못해도 shape 에러가 나지 않고 통과해 버리기 때문입니다.

## 학습 결과 (1 epoch, SGD lr 0.01, H = 16, seed 미고정)

| 모델 | train loss | val loss |
|---|---|---|
| RNN | 2.85 ~ 2.90 | 2.70 ~ 2.75 |
| GRU | 2.98 | 2.80 |

1 epoch 결과이고 seed를 고정하지 않아 실행마다 ±0.05 정도 흔들립니다. 모델 간 비교는 아직 하지 않았습니다.

## 예정

- [ ] NumPy LSTM (forward / backward / torch 비교)
- [ ] RNN / GRU / LSTM 비교 (seed 고정, epoch 늘려서)
- [ ] NumPy predict (npz 불러오기 → 이름 생성)
