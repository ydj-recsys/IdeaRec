# ADRec, BASRec, and SRA-CL: Hyperparameter Configurations and Reproduction Details

## I. Overview of the Three Models and Comparison Conditions

| Item | ADRec | BASRec | SRA-CL |
|---|---|---|---|
| Original paper | *Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach* (KDD 2025) | *Augmenting Sequential Recommendation with Balanced Relevance and Diversity* (AAAI 2025) | *SRA-CL: Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation* (NeurIPS 2025) |
| Method category | Autoregressive, token-level diffusion-based sequential recommendation | A sequential data augmentation plugin that balances relevance and diversity | A contrastive learning enhancement framework based on language-model-driven semantic retrieval |
| Code repository | https://github.com/Nemo-1024/ADRec | https://github.com/KingGugu/BASRec | https://github.com/ziqiangcui/SRA-CL-NeurIPS25 |

## II. ADRec

### 2.1 Experimental Configuration

| Hyperparameter / Procedure | Value or Description Reported in the Paper |
|---|---|
| Embedding dimension and hidden dimension | 128 and 128 |
| Maximum interaction sequence length | 50 |
| Training batch size | 512 |
| Optimizer and learning rate | Adam; `1e-3` |
| Maximum training epochs | 500 |
| Validation interval and early stopping | Validation every 5 epochs; early stopping if the validation metric does not improve over 4 consecutive validation checks |
| Number of diffusion steps | 32 |
| Diffusion noise schedule | Truncated linear |
| Recommendation objective | Full-item cross-entropy loss; no negative sampling is used to compute the recommendation loss |
| Diffusion training and inference | During training, independent noise is applied to each token in the shifted target sequence; during inference, denoising is performed only on the final target position, while historical positions remain noise-free |
| Training strategy | (1) Pretrain item embeddings; (2) freeze the pretrained embeddings and warm up the backbone; (3) jointly fine-tune all parameters |

### 2.2 Default Values in `src/config.yaml`

| Configuration key | Repository default | Notes |
|---|---:|---|
| `hidden_size`, `max_len`, `batch_size` | `128`, `50`, `512` | Consistent with the main configurations reported above |
| `lr`, `weight_decay`, `epochs` | `0.001`, `0.00001`, `500` | Default values in the code |
| `dropout`, `emb_dropout` | `0.1`, `0.3` | Feature and embedding dropout |
| `dif_blocks`, `dif_decoder` | `2`, `att` | Attention-based denoising decoder with 2 blocks |
| `diffusion_steps`, `noise_schedule` | `32`, `trunc_lin` | Number of diffusion steps and noise schedule |
| `beta_a`, `beta_b` | `0.3`, `10` | Schedule parameters in the code |
| `dif_objective`, `schedule_sampler_name` | `pred_x0`, `uniform` | Predicts the clean representation; diffusion time steps are sampled uniformly during training |
| `parallel_ag`, `independent`, `is_causal` | `true`, `true`, `true` | Switches for token-level autoregression, independent diffusion, and causal attention |
| `loss`, `loss_scale` | `mse`, `1.` | Denoising loss and its default scaling configuration; the combined training loss should also be checked against the training code |
| `eval_interval`, `patience` | `5`, `4` | Validation and early stopping |
| `random_seed` | `2025` | Default seed |
| `decay_step`, `gamma` | `100`, `0.1` | The code includes StepLR parameters, whereas the paper reports cosine annealing; the scheduler actually used in the training run must be recorded |

### 2.3 Reproduction Procedure and Details to Be Disclosed in This Study

```bash
cd ADRec/src
python main.py --dataset baby --model pretrain   # If pretrained item embeddings need to be generated locally
python main.py --dataset baby --model adrec      # This entry point can also be used with the officially provided pretrained weights
```

## III. BASRec

### 3.1 Experimental Configuration

| Hyperparameter / Procedure | Value or Description Reported in the Original Paper |
|---|---|
| Item embedding dimension | 64 |
| Maximum sequence length | 50 |
| Batch size | 256 |
| Optimizer | Adam, with a learning rate of `0.001`, `β1=0.9`, and `β2=0.999` |
| Mixup coefficient | `λ ~ Beta(α, α)`; the paper considers `{0.2, 0.3, 0.4, 0.5, 0.6}` as candidate values of `α` for hyperparameter tuning |
| Single-sequence augmentation rate | `rate ~ Uniform(a, b)`; candidate values reported in the paper are `{0.1, 0.2, 0.3}` for `a` and `{0.6, 0.7, 0.8}` for `b` |
| Core augmentation operations | Single-sequence M-Reorder, M-Substitute, and adaptive reweighting; cross-sequence item-wise / feature-wise nonlinear mixup |
| Training procedure | First train the original recommendation backbone, then enable both types of augmentation and their associated loss terms for the second training stage; augmentation modules are disabled during inference |

### 3.2 Configuration and Example for the SASRec Variant

The variant adopted in this study is **BASRec + SASRec**. Its relevant default code settings include `hidden_size=64`, `num_hidden_layers=2`, `num_attention_heads=2`, `max_seq_length=50`, `batch_size=256`, `lr=0.001`, `weight_decay=0`, `epochs=200`, `start_valid=100`, `patience=20`, `hidden_dropout_prob=0.2`, `attention_probs_dropout_prob=0.2`, `beta=0.3`, `rate_min=0.2`, `rate_max=0.51`, `n_pairs=1`, and `n_whole_level=1`.

Example command:

```bash
cd BASRec/SASRec/src
python main.py --data_name=Beauty --model_idx=1 --load_pretrain \
  --beta=0.4 --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 \
  --rate_min=0.2 --rate_max=0.71
```

## IV. SRA-CL

### 4.1 Experimental Configuration

| Hyperparameter / Procedure | Value or Description Reported in the Original Paper |
|---|---|
| Recommendation backbone | Transformer; 2 layers with 2 attention heads per layer |
| Item embedding dimension for recommendation | 64 |
| Maximum sequence length | 20 |
| Batch size | 256 |
| Optimizer / learning rate | Adam / `0.001` |
| Dropout | `0.5` for both the embedding and hidden layers |
| Early stopping | Stop if the validation metric does not improve for 10 consecutive epochs |
| Training objective | `L = L_rec + α L_CS + β L_IS`; `α` and `β` weight the inter-user and intra-user contrastive losses, respectively |
| `α`, `β`, and number of retrieved neighbors `k` | The paper's sensitivity analysis suggests a suitable range of approximately `0.05–0.1` for `α` and `β`, and approximately `10` for `k`; the official example uses `α=0.1, β=0.1, k=10` |
| LLM and text encoder | The original paper uses the **DeepSeek-V3 API** to generate user/item textual semantic information, with generation settings `temperature=0` and `top_p=0.001`; pretrained **SimCSE-RoBERTa** then converts the text into semantic vectors, which are cached in advance and kept fixed during training |

### 4.2 Configuration and Example

In addition to the parameters reported in the paper, the command-line defaults include `train_batch_size=256`, `test_batch_size=1024`, `hidden_size=64`, `inner_size=256`, `epochs=100`, `n_heads=2`, `n_layers=2`, `hidden_dropout_prob=0.5`, `attn_dropout_prob=0.5`, `weight_decay=0`, `stopping_step=10`, `eval_step=1`, `valid_metric=MRR@10`, `seed=2024`, `alpha=0.1`, `beta=0.1`, and `k_num=10`.

The reproduction sequence is: **(1) Prepare the data and prompts → (2) Use the language model to generate user and item descriptions → (3) Convert the text descriptions into semantic vectors and cache them → (4) Train the recommendation model.** Example:

```bash
cd 'SRA-CL-NeurIPS25/recommender_code'
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset=Beauty --train_dir=default --maxlen=20 --device=cuda \
  --alpha=0.1 --beta=0.1 --k_num=10
```
