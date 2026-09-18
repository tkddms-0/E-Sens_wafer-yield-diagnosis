# 6주 프로젝트 플랜 (축소판) — WM-811K 기반 수율 손실 우선순위 도출 및 계통 불량 진단

> **6주 · 6회 모임 · 3인 팀 · 주당 개인 3~4시간 + 모임 2시간** 기준.
> 전공 수업과 병행하는 학습형 연구 프로젝트로, 분류 모델은 최소화하고 **lot 메타데이터 분석에서 공정에 대한 결론**을 내는 데 집중한다.
> 참고: [semiconductor-career-prep / project-a-build-guide.md](https://github.com/serithemage/semiconductor-career-prep/blob/main/content/portfolio-projects/project-a-build-guide.md), [domain-knowledge.md](https://github.com/serithemage/semiconductor-career-prep/blob/main/content/domain-knowledge.md)

---

## 0. 주제 정의

> WM-811K 웨이퍼맵을 결함 유형으로 분류한 뒤, 그 라벨을 **lot · waferIndex · 다이 pass/fail**과 결합해
> (1) 어떤 결함 유형이 수율을 가장 많이 깎는지, (2) 어떤 lot이 계통 불량인지, (3) lot 안 어느 위치에서 결함이 발생하는지를 데이터로 밝힌다.

| 단계 | 하는 일 | 나오는 결론 |
| --- | --- | --- |
| 1. 분류 (CNN) | 웨이퍼맵 → 결함 유형 라벨 | 이후 분석의 입력 (정확도를 겨루지 않음) |
| 2. 수율 손실 Pareto | 결함 유형별 평균 수율 손실 × 발생 건수 | "건수 1위 ≠ 손실 1위" → 개선 우선순위 |
| 3. lot 공통성 지표 | lot 내 웨이퍼 누적맵의 집중도 점수 | 계통 불량 lot 비율, 계통 lot의 수율 손실 배수 |
| 4. lot 내 위치 의존성 | 계통 lot의 waferIndex 구간별 발생률 검정 | 결함 유형별 의심 원인 계열 (누적 열화형 / 슬롯형 / 무관) |

**제목 예시**: 웨이퍼맵 결함 분류와 lot 메타데이터 분석을 통한 수율 손실 우선순위 도출 및 계통 불량 진단

---

## 1. 팀 역할

| 역할 | 담당 영역 | 핵심 기술 |
| --- | --- | --- |
| **M (모델)** | 전처리, 소형 CNN 1회 학습, Grad-CAM 그림, waferIndex 검정 | PyTorch, 불균형 처리, 카이제곱 검정 |
| **S (공간통계)** | 수율 계산, Pareto, lot 누적맵·집중도 지표 | 공간 통계, 확률분포, 임계값 결정 |
| **A (분석·리포트)** | EDA, 도메인 조사, 계통 lot 육안 검토, 리포트·발표 | 반도체 공정 지식, 시각화, 문서화 |

> 매주 모임에서 서로의 코드를 30분씩 리뷰해 전원이 전체 파이프라인을 설명할 수 있게 유지한다.

---

## 2. 범위 (원안 대비 축소)

| 항목 | 원안 | 축소판 |
| --- | --- | --- |
| 분류기 | ResNet-18 전이학습, Grad-CAM, 혼동 분석 | 소형 CNN 1회 학습 (32×32, `none` 다운샘플링), Grad-CAM 그림 1장 |
| 수율 손실 귀속 | KDE vs Poisson 분해 + 합성 검증 | **삭제** → 라벨 웨이퍼 실제 수율로 유형별 평균 손실 × 건수 |
| lot 공통성 지표 | 후보 3개 비교 | **집중도 1개**만 구현 (상위 10% 다이가 차지하는 불량 비율) |
| 외부 검증 | MixedWM38 | 삭제 |
| 문서 | ADR 4건, 리포트 10~15쪽 | ADR 3건, 리포트 6~8쪽 |

향후 과제로 명시: 오토인코더 준지도 사전학습, KDE 기반 다이 단위 귀속, 공통성 지표 후보 비교, 하위 패턴 군집화.

---

## 3. 데이터셋 한계와 대응 ⚠️

WM-811K는 실제 fab 데이터지만 **장비·공정·시간 정보가 모두 익명화**되어 있다. 각 한계를 (a) 우회 방법, (b) 리포트 기술 방식으로 나눠 정리한다.

| # | 한계 | 영향 | (a) 우회 | (b) 리포트에 쓸 문장 |
| --- | --- | --- | --- | --- |
| 1 | **장비·챔버 ID 없음** | 실제 fab의 Commonality 분석(어느 장비를 거쳤나) 불가 | lot 단위 공간 공통성 + waferIndex 의존성을 **장비 이력의 대리 지표**로 사용 | "장비 이력 부재로 원인 공정은 확정할 수 없으며, 본 연구의 원인 추정은 공간 패턴과 lot 내 위치 분포에 근거한 가설 수준이다." |
| 2 | **공정 단계·검사 시점 없음** | 웨이퍼맵이 어느 공정 직후인지 알 수 없음 (최종 EDS 맵으로 추정) | 결함을 **특정 공정이 아니라 원인 계열**(누적 열화형 / 슬롯형 / 무관)로만 분류 | "WBM은 전 공정 누적 결과이므로 특정 단계를 지목하지 않고 문헌 매핑을 가설로 제시한다." |
| 3 | **타임스탬프 없음** | 시간 추세·SPC 불가, lot 순서 불명 | 시간 분석은 범위 밖으로 명시. `lotName` 번호가 순서일 가능성은 있으나 **검증 불가하므로 사용하지 않음** | "lotName의 순서성은 문서화되어 있지 않아 시계열 해석에 사용하지 않았다." |
| 4 | **lot당 웨이퍼 수 불완전** | 25장 미만 lot 다수 → 공통성 지표 불안정 | 1주차에 분포 확인 후 **N장 이상 lot만** 분석 (N은 분포 보고 결정, 예: 10) | "분석 대상 lot을 N장 이상으로 제한했으며, 이는 전체 lot의 X%다." |
| 5 | **제품·기술 노드 불명** | 여러 제품이 섞여 있어 결함 분포가 제품 효과일 수 있음 | `dieSize`를 제품 대리 변수로 사용해 **dieSize 그룹별로 층화**해서 결과 재확인 | "dieSize를 제품 대리 변수로 층화한 결과, 주요 결론이 그룹 간 일관됨/상이함을 확인했다." |
| 6 | **라벨 21%만 존재, 라벨러 정보 없음** | 라벨 노이즈, 무라벨 웨이퍼의 선택 편향 가능 | 라벨 데이터만 사용하고 이를 명시. A 담당이 계통 lot 20~30개 육안 교차 검토로 **라벨 일관성 부분 검증** | "라벨된 17만 장만 사용했으며, 무라벨 63만 장은 향후 과제(준지도 학습)로 남긴다." |
| 7 | **bin 코드 없음 (pass/fail 2값)** | 전기적 불량 모드(open/short/leak) 구분 불가 | 공간 패턴만으로 분석하고 이를 한계로 기술 | "bin 코드 부재로 전기적 불량 모드는 구분하지 않았다." |
| 8 | **waferIndex의 의미 미문서화** | 카세트 슬롯 번호인지 처리 순서인지 불명 | 매엽 공정이면 처리 순서, 배치 공정이면 슬롯 위치 — **두 해석을 병기** | "waferIndex 의존성은 처리 순서 효과(매엽) 또는 슬롯 위치 효과(배치) 두 가지로 해석 가능하다." |
| 9 | **웨이퍼맵 크기 불일치·저해상도** | 리사이즈 시 미세 패턴(Scratch) 손실 | 32×32 리사이즈 전 원본 해상도로 공간 통계 계산, CNN 입력만 리사이즈 | "공간 통계는 원본 해상도, 분류기 입력만 리사이즈했다." |
| 10 | **fab·시기 불명 (2015년 대만 fab 추정)** | 현재 공정과 결함 특성이 다를 수 있음 | 방법론의 일반성을 강조 | "본 방법은 lot·위치 메타데이터가 있는 어떤 WBM 데이터에도 적용 가능하며, 실제 fab에서는 장비 이력과 결합해 확정 진단으로 확장된다." |

> **핵심 논리**: 한계 1·2는 이 프로젝트의 **연구 동기**이기도 하다. 문헌마다 같은 패턴에 다른 원인을 붙이는 이유가 이 정보 부재 때문이고, lot 위치 의존성 검정은 그 가설들을 데이터로 구분하려는 시도다.

---

## 4. 주차별 플랜 + 추천 논문

논문은 **이론·리뷰 위주**로 골랐다. 각 역할이 그 주에 쓸 방법의 근거를 읽는 것이 목적이며, 전부 읽을 필요 없이 "우리 프로젝트에 가져올 것" 항목만 채우면 된다.

### 공통 사전 학습 (1주차 모임 전)

- [ ] 반도체 8대 공정, 웨이퍼 → 다이 → lot, 수율 정의, Wafer Bin Map
- [ ] Wu, Jang, Chen, *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*, IEEE TSM, 2015 — 데이터셋 원논문
- [ ] Batool et al., *A Systematic Review of Deep Learning for Silicon Wafer Defect Recognition*, IEEE Access, 2021 — 분야 전체 조망용 리뷰
- [ ] Kaggle `qingyi/wm811k-wafer-map`에서 `LSWMD.pkl` 다운로드 후 각자 로드

---

### 1주차 — 데이터 이해 · EDA · 수율 계산 (역할 분담 전, 전원 공동)

**모임 목표**: 데이터 구조·lot 크기 분포·수율 분포를 전원이 숫자로 설명

| 역할 | 할 일 | 추천 논문 |
| --- | --- | --- |
| M | pkl 로드, 가변 크기 → 32×32 리사이즈(0/1/2 유지), `lotName` 기준 `GroupShuffleSplit` | Zhang, Lipton, Li, Smola, *Dive into Deep Learning* (무료 온라인) — 3~4장 (선형/MLP), 7장(CNN) 훑기 |
| S | 웨이퍼별 수율 계산, lot당 웨이퍼 수 분포, waferIndex 분포, dieSize 분포 | Stapper, Armstrong, Saji, *Integrated Circuit Yield Statistics*, Proc. IEEE, 1983 — 수율 통계의 기본 |
| A | 패턴별 샘플 시각화, 불균형 막대, 라벨/무라벨 비율, 패턴→원인 문헌 매핑표 1차 | Hansen, Nair, Friedman, *Monitoring Wafer Map Data for Spatially Clustered Defects*, Technometrics, 1997 — 계통 vs 랜덤, 패턴→원인의 원출처 |

- **모임에서 확인**: 25장 미만 lot 비율 → 분석 대상 N 결정, `none` 비율, 수율 분포
- **ADR 1**: 왜 lotName 그룹 분할인가 / 분석 대상 lot 기준 N을 왜 이렇게 정했나

---

### 2주차 — 소형 CNN · 수율 Pareto · 매핑표

**모임 목표**: 분류기 1회 학습 완료, 결함 유형별 수율 손실 표

| 역할 | 할 일 | 추천 논문 |
| --- | --- | --- |
| M | 소형 CNN(Conv 3층) 1회 학습, class-weighted CE, per-class recall·balanced accuracy, 혼동행렬 | He & Garcia, *Learning from Imbalanced Data*, IEEE TKDE, 2009 — 불균형 처리 리뷰; Buda, Maki, Mazurowski, *A Systematic Study of the Class Imbalance Problem in CNNs*, Neural Networks, 2018 |
| S | 결함 유형별 평균 수율 손실 × 건수 Pareto, dieSize 그룹별 층화 재확인 | Cunningham, *The Use and Evaluation of Yield Models in IC Manufacturing*, IEEE TSM, 1990 — Poisson·음이항 모델 |
| A | 패턴→원인 매핑표 확정(논문별 원인 병기), 원류 논문 정리 | Wang, Kuo, Bensmail, *Detection and Classification of Defect Patterns on Semiconductor Wafers*, IIE Trans., 2006; Chen & Liu, *A Neural-Network Approach to Recognize Defect Spatial Pattern in Semiconductor Fabrication*, IEEE TSM, 2000 |

- **모임에서 확인**: 건수 순위 vs 손실 순위 차이 여부 (없어도 결론), 분류기 balanced accuracy
- **ADR 2**: 왜 소형 CNN인가(전이학습 아님), 왜 class-weight인가
- ⚠️ 이후 분류기 수정 금지

---

### 3주차 — lot 누적맵 · 집중도 지표 · 육안 검토

**모임 목표**: 계통 lot 목록과 비율

| 역할 | 할 일 | 추천 논문 |
| --- | --- | --- |
| S | lot 내 웨이퍼맵 누적 → 다이 위치별 불량 빈도 맵 → 집중도 점수(상위 10% 다이의 불량 비율) → 분포 보고 임계값 결정 → 계통 lot 분리 | Moran, *Notes on Continuous Stochastic Phenomena*, Biometrika, 1950 — 공간 자기상관 개념 (향후 후보 비교용); Hsu & Chien, *Hybrid Data Mining Approach for Pattern Extraction from Wafer Bin Map*, IJPE, 2007 |
| M | Grad-CAM 결과 그림 1장(패턴별 대표 1개씩), CNN 결과 정리 | Selvaraju et al., *Grad-CAM*, ICCV, 2017; Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS, 2018 — Grad-CAM 해석 시 주의점 |
| A | 계통 lot 후보 20~30개 육안 검토(전원 교차), 지표 상위 lot과 일치율 | Chien, Wang, Cheng, *Data Mining for Yield Enhancement in Semiconductor Manufacturing*, ESWA, 2007 — 수율 분석 실무 흐름 |

- **모임에서 확인**: 계통 lot 비율, 계통 lot vs 비계통 lot 수율 손실 배수, 육안 일치율
- **ADR 3**: 임계값을 어떻게 정했나, 버린 대안

---

### 4주차 — waferIndex 의존성 검정 · 해석

**모임 목표**: 결함 유형별 "위치 의존성 있음/없음" 표

| 역할 | 할 일 | 추천 논문 |
| --- | --- | --- |
| M | 계통 lot의 결함 유형별 waferIndex(1~25) 발생률 곡선, 전반/중반/후반 카이제곱 검정, Bonferroni 보정 | Agresti, *An Introduction to Categorical Data Analysis*, Wiley — 2~3장(분할표·카이제곱) |
| S | 검정 결과를 집중도 점수와 교차(집중도 높을수록 위치 의존성 강한가), dieSize 층화 재확인 | Getis & Ord, *The Analysis of Spatial Association by Use of Distance Statistics*, Geographical Analysis, 1992 (개념만) |
| A | 결과를 원인 계열(누적 열화형 / 슬롯형 / 무관)로 해석, 매핑표 가설 중 지지·기각 표기 | Chien, Hsu, Chen, *Semiconductor Fault Detection and Classification for Yield Enhancement and Manufacturing Intelligence*, Flex. Serv. Manuf. J., 2013 — FDC·수율 연결 |

- **모임에서 확인**: 유의한 결함 유형, 방향(전반/후반/특정 번호), 가설 기각 항목도 그대로 기록
- 한계 8(waferIndex 의미) 두 해석 병기 확인

---

### 5주차 — 재현 · 리포트 작성

**모임 목표**: 파이프라인 1회 실행으로 전체 표·그림 재생성, 리포트 초안

| 역할 | 할 일 | 추천 자료 |
| --- | --- | --- |
| M | 전체 파이프라인 단일 스크립트(seed 고정), `results/` 자동 생성 확인 | — |
| S | 임계값·N 변경 시 계통 lot 비율 변화(민감도 2~3점만) | — |
| A | 리포트 초안: 문제 → 데이터·한계(3장 표 활용) → 방법 → 결과 → 해석 → 한계·향후 과제 | Mensh & Kording, *Ten Simple Rules for Structuring Papers*, PLoS Comp Bio, 2017; Rougier et al., *Ten Simple Rules for Better Figures*, PLoS Comp Bio, 2014 |

- **모임에서 확인**: 민감도에서 결론이 뒤집히는지, 리포트 목차 합의

---

### 6주차 — 최종 발표 · 정리

| 역할 | 할 일 |
| --- | --- |
| A | 리포트 6~8쪽 완성, 발표 자료 12장 내외 |
| M | README(데이터 출처·실행 순서·환경), 주요 함수 docstring |
| S | 지표 정의 부록, 한계·향후 과제 섹션 |
| 전원 | 리허설, 예상 질의(왜 이 임계값, 장비 정보 없이 원인을 어떻게 말하나, 실제 fab과의 차이) |

- **모임에서 결정**: 학회 포스터·학부 논문 제출 여부, 다음 학기 이어갈 향후 과제

---

## 5. 모임 루틴

| 구간 | 내용 |
| --- | --- |
| 30분 | 진행 공유 |
| 30분 | 코드 리뷰 (PR 단위) |
| 30분 | 결과 해석 토론 |
| 15분 | 다음 주 할 일 확정 → Projects 보드 |

- 주 1건 결정을 `docs/decisions.md`에 기록
- 결과가 예상과 달라도 수정하지 않고 기록
- main 직접 커밋 금지, 역할별 브랜치 → PR → 팀원 1명 리뷰 후 머지

---

## 6. 리스크

| 리스크 | 대응 |
| --- | --- |
| lot당 웨이퍼 수 부족 | 1주차에 N 결정, 대상 lot 비율을 리포트에 명시 |
| GPU 없음 | Colab T4, 32×32, `none` 다운샘플링 — 1 epoch 수 분 수준 |
| 시험 기간과 겹침 | 3주차(집중도 지표)만은 밀리지 않게, 2주차 CNN은 1회 학습으로 고정 |
| 3주차에서 계통 lot이 거의 안 나옴 | 임계값 완화 또는 N 완화, 그래도 없으면 "계통성 약함"을 결론으로 |

---

## 7. 레포 구조

```
E-Sens_wafer-yield-diagnosis/
├── README.md
├── docs/
│   ├── PLAN.md             # 이 문서
│   ├── study/              # 논문 1편 = md 1개 (문제 / 방법 / 가져올 것 / 의문점)
│   ├── meetings/           # week_N.md 회의록
│   ├── decisions.md        # ADR
│   ├── limitations.md      # 3장 한계표 상세판
│   └── report/             # 리포트, 발표 자료
├── src/
│   ├── data/               # 로드, 리사이즈, 그룹 분할 (M)
│   ├── model/              # CNN, Grad-CAM (M)
│   ├── spatial/            # 수율, Pareto, 누적맵, 집중도 (S)
│   └── analysis/           # EDA, 검정, 시각화 (M/A)
├── notebooks/
├── results/                # 재현 스크립트가 생성
├── requirements.txt
└── .gitignore              # LSWMD.pkl, *.pt 제외
```

---

## 체크리스트

- [ ] 1주차: lot 크기 분포 확인, 분석 대상 N 결정, 수율 분포 확인
- [ ] 2주차: CNN 1회 학습·고정, 수율 Pareto 표 완성
- [ ] 3주차: 집중도 지표로 계통 lot 식별, 육안 일치율 기록
- [ ] 4주차: waferIndex 의존성 검정, 가설 지지/기각 표
- [ ] 5주차: 파이프라인 재현, 리포트 초안
- [ ] 6주차: 발표, `decisions.md` 3건, 한계표 리포트 반영
