```mermaid
flowchart LR
  subgraph Warmup[Stage 0: Warmup (仅 IM)]
    A[Target Images] --> B[Encoder]
    B --> C[Classifier]
    C --> D[p_model]
    D --> E[IM Loss: E_H − H(m)]
    E --> F[Optimizer Update]
  end

  subgraph Main[Stage 1: Main (全分支)]
    A2[Target Images] --> B2[Encoder]
    B2 --> C2[Classifier]
    C2 --> D2[p_model]

    subgraph CLIP[CLIP 先验分支]
      T[Class Names] --> P[Prompt]
      P --> TE[Text Encoder]
      TE --> TC[Text Prototypes t_c]
      A2 --> IE[CLIP Image Encoder]
      IE --> GI[Image Features g_i]
      GI --> ZS[Zero-shot Probs p_zs = softmax(β·<g_i,t_c>)]
    end

    D2 --> SOFTFUSE
    ZS --> SOFTFUSE
    SOFTFUSE[p_soft = α·p_model + (1−α)·p_zs] --> SCE[Soft CE (conf^pow)]

    ZS --> KD[KD: KL(p_zs || p_model)]

    subgraph SEL[选择与对比]
      FEAT[(Features: g_i or v_i)]
      FEAT --> KNN[KNN Voting: exp(sim/τ_knn), P_i(c)]
      KNN --> BAL[Balanced Selection: per-class top sel_ratio]
      BAL --> PAIRS[Selected Pairs]
      PAIRS --> SUP[SupCon (no CLIP gating)]
    end

    D2 --> IM[IM: E_H − H(m)]

    SCE --> SUM[Weighted Sum]
    KD --> SUM
    SUP --> SUM
    IM --> SUM
    SUM --> UPD[Optimizer Update]
  end
