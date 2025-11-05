# AI_Based_Drug_Discovery


# 🧬 신약개발 경진대회 [(Link)](https://dacon.io/competitions/official/236518/leaderboard)

#### Leaderboard 1위


---

## 🏆 Result & Highlights

* **Private 리더보드 압도적 1등**
* **Chemical Space 다양성** 극대화
* **Robust GNN+SMILES Mixup** 기반 AI
* **확장성:** DDI 예측 등 범용 분자 AI 솔루션 제안

---

## 🧐 About

* **목표:**
  CYP3A4 저해율(%)을 예측하는 AI 모델 개발
* **CYP3A4?**

  * 전체 약물의 30~50%를 대사하는 **주요 간 효소**
  * 크고 유연한 Active Site → 다양한 구조(Substrate) 대응
  * 저해 발생 시 독성·부작용↑, 신약 개발 필수 평가
* **도전과제:**

  * **비용·시간 많이 드는 전통적 실험 → AI로 대체**
  * 데이터 1,681개로 화학공간(분자 다양성) 한계
  * 평가 지표: **RMSE + Correlation** (정확도 & 변화 추적 모두 중요)

---

## 📦 Data Strategy

* **외부 데이터 수집:**

  * **PubChem, ChEMBL**에서 유사 조건 데이터 확보
  * 벡터 공간(임베딩) 상 **유사성 검증 → 합류**
  * Chemical Space 다양성·외삽력 강화

* **SMILES Mixup (Custom Algorithm):**

  * 기존 Mixup 아이디어 → **SMILES 특화** 확장
  * **임베딩 유사도가 높은 분자끼리** 혼합,
    실제 관측 불가한 구조적 다양성/일반화력 ↑
  * *분자 구조-활성 관계* 학습 개선

---

## 🔬 Feature Engineering

* **구조적 Feature:**

  * **Fingerprint**: 분자 패턴, 구조적 유사성
* **물리화학 Feature:**

  * **Descriptor**: SLogP, FilterLogS, MindO, QED 등

    * **LogP, TPSA**: 극성-소수성 Pocket 특성
    * **전하분포 자기상관, 방향족비율, 작용기**: 결합·대사 특성
    * **Lipinski rule of 5, QED**: 약물성 규칙
* **공간적 Feature:**

  * **SMILES 임베딩**: 구조적 패턴·Similarity 효과적 학습
* **Feature Importance 분석:**

  * SLogP, FilterLogS, QED, 전하분포 등 실제 모델 핵심

---

## 📁 Validation & Experiment Design

* **Random K-Fold 검증 채택**

  * **Stratified/Scaffold K-Fold**: 노이즈/불균형 유발
  * Random K-Fold: 유사 구조 **Fold 간 분산** → **현실성/강건성**↑
  * Validation Score 편차 관리로 **강건성 보증**
* **외부데이터/데이콘 데이터 성능 동등**

  * 외부 데이터 사용 적합성 검증 완료

---

## 🤖 Modeling Strategy

### **메인 모델:**

**Graph Neural Network (GNN) 기반 분자 그래프 모델**

* **Atom/Bond 특징 임베딩**
* **링 구조, 고리 소속 정보** 적극 활용 → 구조 안정성까지 표현
* **2단계 학습**

  1. **사전학습:** Graph Contrastive Learning

     * **Node Masking** 등으로 분자 일반화 능력↑
  2. **파인튜닝:** CYP3A4 저해율 예측
* **Test Time Augmentation (TTA)**

  * 하나의 분자 SMILES → 여러 순서로 증강,
    모델이 특정 표기 의존↓, **일반화 성능↑**

### **실험 및 최적화**

* **Custom Loss:**

  * 표준편차 기반 Loss로 예측 다변성 확보(중앙 집중 경향 해소)
* **SMILES 순서 증강:**

  * 다양한 SMILES 순서로 표현, 표현 다양성/강건성↑
* **Multimodal:**

  * **SMILES + FingerPrint Dual Input**
* **라벨 노이즈 주입:**

  * 실험 환경 변동 반영, 불확실성 학습 유도
* **Feature 조합 최적화:**

  * 구조→물리화학→공간특성→TTA 순 적용, 최종 0.78 Correlation 달성
* **GNN + Mixup** 조합: 단독 0.75 → 0.78↑ (Ensemble 효과)

---

## ⚡ 실험 관리

* **다양한 Loss/데이터/증강/입력 조합 실험**
* **DACON 데이터 단독 Validation** → 외부/내부 성능 차이 無

---

## 🔗 Generalization & Application

* **확장성:**

  * CYP3A4 외 **대사 관련 예측** (예: CYP-기질 친화도 등)
  * **Drug-Drug Interaction(DDI)** 예측

    * **Drug Pair 입력**에 솔루션 확장
    * **SMILES Mixup** 활용 →

      1. 약물쌍 불균형
      2. SMILES 순서 민감성
      3. 일반화
      4. 다양성 부족
         모두 개선 가능
  * **분자 생성 모델 입력 증강**에도 활용
* **DDI 특화**

  * Mixup: 소수 클래스/다수 클래스 균형, 벡터기반이므로 SMILES 순서 무의미화,
    **새로운 분자 표현** 대량 생성해 Unseen 대응력 강화

---

## 📂 Dataset & 주요 변수

* **Target:** CYP3A4 저해율 (%)
* **Input:**

  * SMILES, FingerPrint, Descriptor(물리화학적 특성),
    임베딩, 작용기/방향족비율/전하분포 등
  * 외부 데이터(PubChem, ChEMBL), SMILES Mixup 샘플

---

## 🎈 Key Modeling Summary

```text
Data
  - DACON + 외부 데이터(PubChem, ChEMBL) + SMILES Mixup 증강
Feature
  - 구조(FP), 물리화학(Descriptor), 공간(SMILES 임베딩), Domain 특화 Feature
Model
  - Graph Neural Network(GNN), Multimodal(SMILES + FP), Custom Loss, TTA
Validation
  - Random K-Fold, Score 편차 관리
Application
  - DDI/대사 예측/분자 생성 등 범용성
```


