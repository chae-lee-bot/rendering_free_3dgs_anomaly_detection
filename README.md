# 렌더링 없는 3D 이상 탐지 (3D Gaussian Splatting 기반)

> **3D-to-3D 이상 탐지 프레임워크** — 2D 렌더링 없이 14차원 Gaussian 파라미터 공간에서 직접 이상 탐지를 수행합니다.

---

## 프로젝트 요약 (Project Summary)

본 연구는 **3D Gaussian Splatting(3DGS)** 기반의 산업 이상 탐지 프레임워크를 제안합니다. 기존 방법들이 3D 표현을 2D 이미지로 렌더링한 후 탐지를 수행하는 것과 달리, 본 프레임워크는 3D Gaussian 공간에서 직접 이상을 탐지하여 렌더링으로 인한 시간을 단축시키고 정보 손실을 방지하고자 합니다.

### 핵심 기여

- **렌더링 없는 탐지**: `xyz, f_dc, opacity, scale, rotation` 등 14차원 Gaussian 파라미터에서 직접 이상 스코어를 계산합니다.
- **MAE 재구성 브랜치**: Point-MAE 기반의 마스크 오토인코더로, FPS+KNN Tokenization(`G=1024, K=32`, 마스킹 비율 60%)을 적용합니다. 추론 시 반복 마스킹(`n_iter=40`)으로 안정적인 이상 스코어를 획득합니다.
- **KDTree 밀도 브랜치**: 정상/테스트 포인트 클라우드 간 지역 밀도를 비교하여 구조적 이상(버르, 누락)을 탐지합니다. 적응 반경(`r = 1.5 × 평균 13-NN 거리`)을 사용합니다.
- **하이브리드 게이팅**: 샘플별 기하 통계(ratio, concentration, sharpness)를 기반으로 MAE 스코어와 밀도 스코어 중 하나를 선택합니다.

### 데이터셋

**MAD-Sim** 데이터셋 — 20개 LEGO 클래스, 3가지 이상 유형:

| 이상 유형 | 설명 |
|---|---|
| **Burrs** | 표면에 돌출된 여분의 재료 |
| **Stains** | 표면의 색상/텍스처 오염 |
| **Missing** | 부품 누락 |

---

## 실행 방법 (Code Instruction)

### 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/chae-lee-bot/rendering_free_3dgs_anomaly_detection.git
cd rendering_free_3dgs_anomaly_detection

# 가상환경 생성 및 활성화
conda create -n 3dgs-anomaly python=3.9
conda activate 3dgs-anomaly

# 패키지 설치
pip install -r requirements.txt
```

### 디렉토리 구조

```
rendering-free-3dgs-anomaly-detection/
├── model/          # MAE 재구성 모델 (Point-MAE 기반)
├── density/        # KDTree 밀도 스코어링
├── gating/         # 하이브리드 게이팅 (compute_gate_v3/v4)
├── sbatch/         # SLURM 실행 스크립트
└── README.md
```

### 데이터 준비


```
data/
├── MAD_SIM_FINAL/
│   └── {class}/
│       ├── normal/point_cloud_*.ply
│       └── anomaly/{type}_recon/random_*/point_cloud.ply
├── MulSen_3DGS/         
└── Anomaly_ShapeNet/    
```

> 데이터 다운로드: [Google Drive] https://drive.google.com/drive/folders/1bFZRKoUekXcvtJZ2oppDKeiyez4hu1yx?usp=sharing

### 학습

```bash
python train.py --config configs.py --dataset MAD_SIM_FINAL --output output/mae_final.pt
```

### 평가

```bash
# 하이브리드 게이팅 평가
python compute_gate_v4.py \
    --mae_scores eval_results/eval_raw_orig.npz \
    --density_scores density_bidir.npz \
    --q 0.35

```

---

## 실험 결과 (Demo)

### 정량적 결과 (Per-Gaussian AUROC)



<img width="351" height="95" alt="image" src="https://github.com/user-attachments/assets/fb18adfb-02bc-4123-91d3-214cd9263781" />



---

## 결론 및 향후 연구 (Conclusion and Future Work)

### 결론

본 연구는 3D Gaussian Splatting 표현을 직접 처리하는 **렌더링 없는 3D 이상 탐지 프레임워크**를 제안합니다. MAE 재구성 브랜치와 KDTree 밀도 브랜치를 하이브리드 게이팅으로 결합하여 MAD-Sim에서 전체 AUROC **0.792**를 달성했습니다. 

이상 유형별로 최적의 탐지 전략이 다름을 확인했습니다: 외관 이상(stains)은 재구성 오류 기반 탐지에, 구조적 이상(missing, burrs)은 밀도 기반 탐지에 더 적합합니다. 하이브리드 게이팅은 이 간극을 효과적으로 해소합니다.

### 향후 연구

- **속성별 디코더**: 기하(`xyz`, `scale`, `rotation`)와 외관(`f_dc`, `opacity`) 파라미터를 분리 디코딩하여 세밀한 이상 스코어링
- **MAE + 메모리 뱅크 하이브리드**: 정상 Gaussian 패턴 메모리 뱅크를 결합하여 일반화 성능 향상
- **실제 산업 데이터셋**: 합성 LEGO 씬을 넘어 실제 산업 환경으로의 확장

---

## 참고문헌

- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)
- [MAD-Sim / PAD](https://github.com/EricLee0224/PAD)
- [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) (Wenqiao Li et al., CVPR 2024)
- [MulSen-AD](https://github.com/hito0448/MulSen-AD) (Wenqiao Li et al., CVPR 2025)
