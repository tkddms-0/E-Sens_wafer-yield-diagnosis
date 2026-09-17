# 6주 프로젝트 플랜 — WM-811K 기반 수율 손실 진단 및 계통 불량 식별

> 이 문서는 **6주 · 6회 모임 · 3인 팀** 기준의 학습형 연구 프로젝트 플랜이다.
> 목표는 웨이퍼맵 분류에서 끝나지 않고, **lot 메타데이터와 다이 단위 수율을 결합해 공정에 대한 유의미한 결론**을 내는 것이다.
> 참고: [semiconductor-career-prep / project-a-build-guide.md](https://github.com/serithemage/semiconductor-career-prep/blob/main/content/portfolio-projects/project-a-build-guide.md), [domain-knowledge.md](https://github.com/serithemage/semiconductor-career-prep/blob/main/content/domain-knowledge.md)

---

## 0. 프로젝트 한 줄 정의

> WM-811K 웨이퍼맵을 CNN으로 분류한 뒤, **다이 단위 수율 손실 귀속**과 **lot 공간 공통성 지표**를 통해 결함 유형별 개선 우선순위와 계통 불량 lot을 데이터로 도출한다.

- **데이터**: WM-811K (Kaggle `qingyi/wm811k-wafer-map`, CC0) — 811,457장, 9패턴, 라벨 약 21%
- **활용 메타데이터**: `waferMap`(다이 pass/fail), `lotName`, `waferIndex`(lot 내 1~25), `dieSize`
- **차별점**: 기존 포트폴리오 가이드의 패턴→원인 매핑은 문헌 기반 정적 표. 본 프로젝트는 그 표를 **가설**로 두고 lot·웨이퍼 위치 데이터로 **검증**한다.

---

## 1. 팀 역할

| 역할 | 담당 영역 | 핵심 기술 |
| --- | --- | --- |
| **M (모델)** | 데이터 전처리, CNN 분류기, Grad-CAM, 검증 세트 생성 | PyTorch, 이미지 분류, 불균형 처리 |
| **S (공간통계)** | 다이 수율 계산, 수율 손실 귀속, lot 공통성 지표 | 공간 통계, 확률분포, 가설검정 |
| **A (분석·리포트)** | EDA, 도메인 조사, 결과 해석, 리포트·발표 | 반도체 공정 지식, 시각화, 문서화 |

> 역할은 고정하되, 매주 모임에서 서로의 코드를 30분씩 리뷰해 **전원이 전체 파이프라인을 설명할 수 있는 상태**를 유지한다.

---

## 2. 범위

| 포함 | 제외 (향후 과제) |
| --- | --- |
| CNN 분류기 + Grad-CAM | 오토인코더 준지도 사전학습 (무라벨 63만 장) |
| 다이 단위 수율 손실 귀속 (핵심 기여 1) | 하위 패턴 군집화 |
| lot 공간 공통성 지표 (핵심 기여 2) | — |
| MixedWM38 외부 검증 (5주차, 여유 시) | — |
| 리포트 · 발표 · `decisions.md` | — |

---

## 3. 공통 사전 학습 (1주차 모임 전)

- [ ] 반도체 8대 공정, 웨이퍼 → 다이 → lot 단위, 수율 정의, Wafer Bin Map 개념
- [ ] Wu, Jang, Chen, *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*, IEEE TSM, 2015 — WM-811K 원논문
- [ ] Kaggle에서 `LSWMD.pkl` 다운로드 후 각자 로드해 컬럼 확인

---

## 4. 주차별 플랜

### 1주차 — 데이터 이해 · EDA · 베이스라인

**모임 목표**: 데이터 구조를 전원이 숫자로 설명할 수 있는 상태

| 역할 | 할 일 | 공부 / 논문 |
| --- | --- | --- |
| M | 가변 크기 웨이퍼맵 64×64 정규화(0/1/2 인코딩 유지), `lotName` 기준 `GroupShuffleSplit`, flatten + RandomForest 베이스라인 → balanced accuracy | `GroupShuffleSplit`, 데이터 누수, balanced accuracy vs accuracy |
| S | 웨이퍼별 수율 = 정상 다이 / (정상+불량) 계산 함수, lot당 웨이퍼 수·`waferIndex` 분포, 결함 유형별 평균 수율 표 | Cunningham, *The Use and Evaluation of Yield Models in IC Manufacturing*, IEEE TSM, 1990 (Poisson·음이항 부분) |
| A | 패턴별 샘플 5장 시각화, 클래스 불균형 막대, 라벨/무라벨 비율, 패턴→의심 공정 매핑표 1차 | domain-knowledge.md 모듈 1·2; Chien, Wang, Cheng, *Data Mining for Yield Enhancement in Semiconductor Manufacturing*, ESWA, 2007 |

- **모임에서 확인**: lot당 웨이퍼 수 25장 미만 lot 비율 (공통성 지표 적용 범위 결정), `none` 비율, 베이스라인 점수
- **ADR 1**: 왜 그룹 분할인가, 왜 이 리사이즈 방식인가

---

### 2주차 — CNN 분류기 · Grad-CAM

**모임 목표**: 분류기 v1 고정, 모델이 실제 불량 영역을 보는지 확인

| 역할 | 할 일 | 공부 / 논문 |
| --- | --- | --- |
| M | ResNet-18 전이학습(1채널 입력), class-weighted CE, per-class recall·macro-F1·혼동행렬, Grad-CAM 부착 | Selvaraju et al., *Grad-CAM*, ICCV 2017; Nakazawa & Kulkarni, *Wafer Map Defect Pattern Classification and Image Retrieval Using CNN*, IEEE TSM, 2018 |
| S | 불량 다이 공간 통계 함수 — 반경별 밀도, 각도 분포, 중심/가장자리 비율, 중심 좌표 정규화 | 극좌표 변환; Jeong, Kim, Lee, *Automatic Identification of Defect Patterns Using Spatial Correlogram and DTW*, IEEE TSM, 2008 |
| A | 오분류 샘플 + Grad-CAM 갤러리, 혼동 쌍(Edge-Loc↔Edge-Ring, Loc↔Random) 분석 초안 | 불균형 평가 지표; Lin et al., *Focal Loss*, ICCV 2017 (class-weight 대안 비교 근거) |

- **모임에서 확인**: Grad-CAM이 Edge-Ring에서 가장자리를 보는지, 혼동 쌍이 공간적으로 왜 유사한지
- **ADR 2**: 왜 CNN(ViT 아님), 왜 class-weight(SMOTE 아님), 버린 시도
- ⚠️ **이후 분류기 수정 금지** — 3·4주차가 핵심이므로 분류 정확도에 시간을 재투자하지 않는다

---

### 3주차 — 다이 단위 수율 손실 귀속 ⭐ 핵심 기여 1

**모임 목표**: "건수 순위 vs 다이 손실 순위" 비교 표 1개

| 역할 | 할 일 | 공부 / 논문 |
| --- | --- | --- |
| S | 웨이퍼별 불량 다이 KDE → 균일 Poisson 기대 밀도(총 불량 수 / 유효 다이 수)와 비교 → 초과 밀도 = 계통 성분, 나머지 = 랜덤 성분 → 결함 유형별 계통 손실 합계로 Pareto | KDE 대역폭(Scott/Silverman), 공간 Poisson 과정, `scipy.stats.gaussian_kde`; Hansen, Nair, Friedman, *Monitoring Wafer Map Data for Spatially Clustered Defects*, Technometrics, 1997 |
| M | 합성 혼합 웨이퍼 생성기 — 단일 결함 웨이퍼 2장을 다이 단위 OR로 합성, 정답 귀속 비율을 아는 검증 세트 100장 | 합성 시 다이 크기 일치·겹침 처리 |
| A | 건수 기준 vs 다이 손실 기준 Pareto 나란히 시각화, 순위가 바뀐 결함 유형 해석 초안 | Pareto 분석, 수율 손실 → 원가 환산 논리 (domain-knowledge 모듈 2.1) |

- **모임에서 확인**: 합성 세트 귀속 오차, 실제 데이터에서 순위 변동 여부 — **변동 없음도 결론으로 기록**
- **ADR 3**: 왜 KDE·Poisson 기준인가, 대역폭 선택 근거

---

### 4주차 — Lot 공간 공통성 지표 ⭐ 핵심 기여 2

**모임 목표**: 계통 lot 식별 + `waferIndex` 의존성 그래프

| 역할 | 할 일 | 공부 / 논문 |
| --- | --- | --- |
| S | lot 내 웨이퍼맵 누적 → 다이 위치별 불량 빈도 맵 → 공통성 점수 후보 3개 구현 (① 웨이퍼 쌍별 공간 상관 평균 ② 누적맵 Moran's I ③ 누적맵 집중도) → 1개 채택, 임계값으로 계통 lot 분리 | 공간 자기상관(Moran's I), 임계값 결정(elbow / 분위수); Wang, Kuo, Bensmail, *Detection and Classification of Defect Patterns on Semiconductor Wafers*, IIE Trans., 2006 |
| M | 계통 lot의 결함 유형별 `waferIndex`(1~25) 발생률 곡선, lot 전반/중반/후반 발생률 차이 카이제곱 검정 | `scipy.stats.chi2_contingency`, 다중 비교 보정(Bonferroni) |
| A | 계통 lot 20~30개 육안 검토 라벨링(전원 교차 검토) → 지표 상위 lot과 일치율, 패턴→공정 매핑표의 "구분 단서"와 대조 | Hsu & Chien, *Hybrid Data Mining Approach for Pattern Extraction from Wafer Bin Map*, IJPE, 2007 |

- **모임에서 확인**: 계통 lot 비율, 계통 lot 수율 손실 배수, `waferIndex` 의존성이 있는 유형 / 없는 유형 — **가설 기각도 그대로 기록**
- **ADR 4**: 지표 후보 3개 중 선택 이유, 임계값 결정 방식

---

### 5주차 — 통합 · 보강 · 해석

**모임 목표**: 파이프라인 1회 실행으로 전체 결과 재현, 해석 확정

| 역할 | 할 일 | 공부 / 논문 |
| --- | --- | --- |
| M | 전체 파이프라인 단일 스크립트화(seed 고정), 재현 확인. 여유 시 MixedWM38 단일 결함 서브셋 일반화 확인 | (선택) Kyeong & Kim, *Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using CNN*, IEEE TSM, 2018 |
| S | 민감도 분석 — KDE 대역폭 변경 시 Pareto 순위 안정성, 공통성 임계값 변경 시 계통 lot 비율 변화 | 민감도 분석 보고 방식 |
| A | 리포트 초안 — 문제 정의 → 방법 → 결과 → 해석 → 한계(장비 ID 부재, 원인 공정은 가설) → 향후 과제 | 학술 보고서 구조, 그림·표 번호 규칙 |

- **모임에서 확인**: 민감도 분석에서 결론이 뒤집히는 구간이 있는지, 리포트 목차 합의

---

### 6주차 — 최종 발표 · 정리

**모임 목표**: 발표 + 논문화 여부 결정

| 역할 | 할 일 |
| --- | --- |
| A | 리포트 완성(10~15쪽), 발표 자료 15장 내외 |
| M | 레포 README(데이터 출처·실행 순서·환경), 주요 함수 docstring |
| S | 지표 정의·수식 부록, 한계와 향후 과제 섹션 |
| 전원 | 발표 리허설, 예상 질의(왜 KDE인가, 왜 이 임계값인가, 실제 fab 데이터와의 차이) |

- **모임에서 결정**: 학회 포스터·학부 논문 제출 여부, 다음 학기 진행 항목

---

## 5. 주차별 공통 루틴

| 구간 | 내용 |
| --- | --- |
| 30분 | 진행 공유 |
| 30분 | 코드 리뷰 (PR 단위) |
| 30분 | 결과 해석 토론 |
| 15분 | 다음 주 할 일 확정 → Projects 보드 This Week로 이동 |

- 매주 결정사항 1건 이상 `docs/decisions.md`에 기록 (왜 이 방식, 버린 대안)
- 결과가 예상과 달라도 수정하지 않고 기록 — 3·4주차 핵심 결과가 "차이 없음"이면 그것이 결론
- main 직접 커밋 금지, 역할별 브랜치 → PR → 팀원 1명 리뷰 후 머지

---

## 6. 리스크와 대응

| 리스크 | 대응 |
| --- | --- |
| lot당 웨이퍼 수 부족 | 1주차에 분포 확인 후 공통성 지표 적용 대상을 N장 이상 lot으로 제한 |
| GPU 부족 | `none` 클래스 다운샘플링, 이미지 32×32 축소 |
| S 담당 부하 집중 | 3주차부터 M이 검증 세트 생성으로 S 지원 |
| 3주차 귀속 결과가 건수 순위와 동일 | "차이 없음"을 결론으로 보고, 4주차에 비중 이동 |

---

## 7. 레포 구조

```
wafer-yield-diagnosis/
├── README.md
├── docs/
│   ├── study/          # 주차별 스터디 노트 (논문 1편 = md 1개)
│   ├── meetings/       # 회의록 week_N.md
│   ├── decisions.md    # ADR
│   └── report/         # 최종 리포트, 발표 자료
├── src/
│   ├── data/           # 로드, 전처리, 그룹 분할 (M)
│   ├── model/          # CNN, Grad-CAM (M)
│   ├── spatial/        # 수율 계산, KDE 귀속, 공통성 지표 (S)
│   └── analysis/       # EDA, 시각화, Pareto (A)
├── notebooks/          # 탐색용, 확정 후 src로 이관
├── results/            # 재현 스크립트가 생성하는 표·그림
├── requirements.txt
└── .gitignore          # LSWMD.pkl, 모델 가중치 제외
```

> 스터디 노트 형식: **문제 정의 / 방법 / 우리 프로젝트에 가져올 것 / 의문점** 4항목 고정.

---

## 체크리스트

- [ ] 1주차: 데이터 구조·불균형·lot 크기 분포를 숫자로 파악했다
- [ ] 2주차: 분류기 v1을 고정하고 Grad-CAM이 실제 불량 영역을 보는지 확인했다
- [ ] 3주차: 건수 기준 vs 다이 손실 기준 Pareto를 비교했다
- [ ] 4주차: 계통 lot을 식별하고 `waferIndex` 의존성을 검정했다
- [ ] 5주차: 파이프라인 1회 실행으로 전체 결과를 재현했다
- [ ] 6주차: 리포트·발표를 완료하고 `decisions.md`에 4건 이상 기록했다
