# Parkinson's Motor Assessment Dashboard

**스마트워치 센서 기반 정량적 운동 분석 · Clinical Decision Support**

파킨슨병 환자를 진단·평가하는 의사를 위한 Dash + Plotly 기반 interactive 대시보드입니다.
스마트워치(가속도계/자이로스코프) 센서 데이터에서 추출한 movement feature를 정량적으로 시각화하여, 의사가 더 정확하고 일관된 판단을 내릴 수 있도록 지원합니다.

---

## 1. 프로젝트 목표

이 대시보드는 **AI 자동 진단 시스템이 아닙니다.**

의사가 이미 알고 있는 임상적 판단을 **정량적 근거로 뒷받침**하는 도구입니다.

| 기존 평가 방식 | 이 대시보드가 제공하는 것 |
|---|---|
| "떨림이 좀 있는 것 같다" | Tremor power: 0.72 (상위 92 percentile) |
| "리듬이 불규칙한 것 같다" | Rhythm irregularity CV: 0.38 |
| "지난번보다 나빠진 것 같다" | Tremor +15.2%, Amplitude -8.3% (방문 대비) |
| "UPDRS 몇 점으로 할까" | Sensor 기반 추정: 3.4 Finger Tapping → Est. 2.8/4 |

---

## 2. 기술 스택

| 영역 | 도구 |
|---|---|
| Dashboard Framework | **Dash** (by Plotly) |
| Visualization | **Plotly Express** + **Plotly Graph Objects** |
| Backend | Dash callback 기반 Python backend |
| Data Processing | pandas, numpy, scipy |
| Data Storage | Demo CSV / JSON |
| Server | Dash 내장 Flask server |

### Full-Stack 구조

```
┌──────────────────────────────────────────┐
│           Frontend Layer                 │
│  Dash Layout + Plotly Interactive Charts  │
│  dcc.Dropdown, dcc.Tabs, dcc.Graph       │
├──────────────────────────────────────────┤
│           Backend Layer                  │
│  Python Callbacks                        │
│  Feature Engineering (CV, Asymmetry 등)  │
│  UPDRS Score Estimation                  │
│  Normative Z-score 계산                   │
├──────────────────────────────────────────┤
│           Data Layer                     │
│  Demo CSV / JSON 파일                     │
│  → 추후 실제 PADS/TULIP 데이터로 교체      │
└──────────────────────────────────────────┘
```

---

## 3. 실행 방법

### 요구 사항

- **Python 3.9 이상** (Python 버전 확인: `python3 --version`)
- pip (Python 패키지 관리자)

### Step 1. 저장소 클론

```bash
git clone https://github.com/sungmoonie/visual-consensus-dashboard.git
cd visual-consensus-dashboard
```

### Step 2. 가상환경 생성 (권장)

프로젝트별 패키지 충돌을 방지하기 위해 가상환경 사용을 권장합니다.

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS / Linux:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
.\venv\Scripts\activate.bat
```

> 활성화되면 터미널 앞에 `(venv)`가 표시됩니다.

### Step 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### Step 4. 데모 데이터 생성 (최초 1회)

```bash
python3 generate_demo_data.py
```

> `data/` 폴더에 CSV/JSON 파일이 생성됩니다. 이미 생성한 경우 이 단계를 건너뛰어도 됩니다.

### Step 5. 대시보드 실행

```bash
python3 app.py
```

실행에 성공하면 아래와 같은 메시지가 출력됩니다:

```
Dash is running on http://127.0.0.1:8050/
```

### Step 6. 브라우저에서 접속

브라우저를 열고 아래 주소로 접속합니다:

👉 **http://127.0.0.1:8050/**

> 종료하려면 터미널에서 `Ctrl + C`를 누르세요.

### 문제 해결 (Troubleshooting)

| 증상 | 원인 | 해결 방법 |
|------|------|----------|
| `command not found: python` | macOS/Linux에서는 `python3`이 기본 | `python` 대신 `python3` 사용 |
| `Address already in use (Port 8050)` | 이전 실행이 아직 종료되지 않음 | `lsof -i :8050`으로 PID 확인 후 `kill <PID>` |
| `ModuleNotFoundError` | 패키지 미설치 또는 가상환경 미활성화 | `pip install -r requirements.txt` 재실행, 가상환경 활성화 확인 |
| `No such file or directory: data/...` | 데모 데이터 미생성 | `python3 generate_demo_data.py` 실행 |

---

## 4. 프로젝트 폴더 구조

```
visual-consensus-dashboard/
├── app.py                     ← 앱 실행 + callback 정의
├── generate_demo_data.py      ← 데모 데이터 생성 스크립트
├── requirements.txt           ← 필요 패키지
│
├── data/                      ← 데모 데이터 파일
│   ├── patients.csv           (환자 5명 기본 정보)
│   ├── task_features.csv      (과제별 특징 점수 150행)
│   ├── left_right_features.csv (좌우 비교 120행)
│   ├── timeseries.csv         (시계열 센서 데이터 6000행)
│   ├── normative_stats.csv    (건강대조군 통계 30행)
│   ├── patient_history.csv    (방문별 경과 기록 90행)
│   └── motion_trail.json      (움직임 궤적 30개)
│
├── src/                       ← 백엔드 모듈
│   ├── data_loader.py         (데이터 로드 함수)
│   ├── feature_engineering.py (지표 계산 함수)
│   ├── figures.py             (차트 생성 함수 14개)
│   └── layout.py              (대시보드 레이아웃 정의)
│
└── assets/
    └── style.css              (의료 대시보드 스타일)
```

---

## 5. 사용 데이터셋

### 5.1 PADS (Parkinson's Disease Smartwatch Dataset)

- **출처**: PhysioNet (`parkinsons-disease-smartwatch/1.0.0`)
- **참여자**: 469명 (PD 환자, 감별진단군, 건강대조군)
- **센서**: 양손 Apple Watch Series 4 (가속도계 + 자이로스코프, 100Hz)
- **Motor Task 11가지**: Resting, Arm lifting, Finger touching, Gait 등
- **활용**: 정량적 motor profile 생성, 좌우 비대칭 분석, rhythm/amplitude 분석

### 5.2 TULIP (Three-dimensional Understanding and Learning of Impairments in Parkinson's)

- **출처**: Zenodo (record 14262223)
- **참여자**: 11명 (PD 환자 + 건강대조군)
- **영상**: 6대 카메라 멀티뷰 synchronized 영상
- **활용**: 영상 기반 keypoint trajectory, 3D 움직임 복원 (추후 확장)

### 5.3 데모 데이터

실제 데이터셋 없이도 대시보드 구조를 확인할 수 있도록 realistic한 demo data를 포함합니다.

| 환자 ID | Group | 나이 | 성별 | 특징 |
|---------|-------|------|------|------|
| P001 | PD | 68 | F | 전형적 PD, Finger Tapping abnormal |
| P002 | PD | 61 | M | 경증 PD, Gait 위주 이상 |
| P003 | Healthy | 59 | F | 건강대조군 (비교 기준) |
| P004 | Differential | 65 | M | 감별진단 (본태성 떨림 등) |
| P005 | PD | 72 | M | 중증 PD, 전반적 이상 |

---

## 6. 대시보드 탭 상세 설명

### 6.1 환자 Overview

**목적**: 선택한 환자의 전반적인 motor/non-motor 상태를 한눈에 파악

**구성 요소**:
- **인구통계 카드**: Patient ID, 나이, 성별, 손잡이, 진단 그룹
- **요약 카드 6개**: Motor Score, Asymmetry, Tremor, Rhythm Irregularity, Non-Motor Score, Top Abnormal Task
- **Radar Chart**: 5개 movement feature (Tremor, Amplitude 감소, Rhythm 불규칙, 좌우 Asymmetry, Instability)의 프로파일을 한눈에 표시
- **Abnormal Task 순위**: 가장 문제되는 motor task 상위 3개를 bar chart로 표시
- **해석 문구**: 자동 생성된 임상 요약 (예: "이 환자는 Finger Tapping에서 가장 높은 abnormality를 보이며, 특히 Rhythm 불규칙 항목이 두드러집니다.")

**임상 활용**: 진료 시작 시 환자의 전체 상태를 빠르게 파악하는 "첫 화면"

---

### 6.2 Task-Feature Heatmap

**목적**: 어떤 motor task에서 어떤 movement feature가 abnormal한지 정밀 분석

**시각화**: 6개 task(행) × 5개 feature(열)의 heatmap
- 행: Finger Tapping, Hand Open/Close, Rest Tremor, Gait, Toe Tapping, Touch Nose
- 열: Tremor, Amplitude 감소, Rhythm 불규칙, 좌우 Asymmetry, Instability
- 색상: 파랑(정상) → 빨강(비정상), 0~1 스케일
- **셀 클릭**: 해당 task-feature 조합의 상세 정보 (score, percentile, severity level) 표시

**임상 활용**: "전체적으로 나쁜가?"보다 **"어떤 동작에서 어떤 종류의 문제가 있는가?"**를 파악

---

### 6.3 좌우 비교

**목적**: 선택한 motor task에서 좌/우 사지의 movement feature 차이를 정량 비교

**시각화**: Mirror Plot (양방향 수평 bar chart)
- 왼쪽(파란색) ← | → 오른쪽(빨간색)
- Feature별 좌우 값과 Asymmetry Index 표시
- **탭 내부 task selector**로 task 변경 가능

**핵심 지표**: `Asymmetry Index = |Left - Right| / ((Left + Right) / 2)`

**임상 활용**: PD는 보통 한쪽 사지가 더 심하게 나타나므로 (lateralized impairment), 비대칭 정도가 진단의 핵심 단서

---

### 6.4 Rhythm & Amplitude

**목적**: 반복 운동에서 bradykinesia의 핵심 지표를 정량 분석

**시각화 3개**:
1. **Amplitude 시계열**: 좌/우 amplitude over time + detected peak markers
2. **Tap Interval 분포**: 반복 간격의 일정성 (초록=일정, 빨강=불규칙)
3. **Amplitude Decrement**: 피크 amplitude가 반복에 따라 감소하는 정도 + trend line

**핵심 지표**:
- `Rhythm Irregularity (CV)` = std(intervals) / mean(intervals)
- `Amplitude Decrement` = (early peaks 평균 - late peaks 평균) / early peaks 평균

**임상 활용**: "불규칙해 보인다", "점점 작아지는 것 같다"는 주관적 판단을 숫자로 확인

---

### 6.5 Normative 비교

**목적**: 환자가 healthy control 대비 어디에 위치하는지 정량 비교

**시각화**: 수평 chart
- 건강대조군 분포 범위 (95% band) 표시
- 건강대조군 평균 (초록 마커)
- 환자 값 (다이아몬드 마커 + Z-score 표시)
- Z > 2: 빨강 (clinically significant), Z > 1: 주황, Z ≤ 1: 파랑

**임상 활용**: "떨림이 좀 있다"가 아니라 "같은 연령대 건강인의 상위 92%에 해당하는 tremor"로 객관적 기준 제공

---

### 6.6 경과 추적

**목적**: 방문별 motor feature 변화를 추적하여 disease progression과 치료 반응 평가

**시각화 2개**:
1. **추이 그래프**: 모든 feature의 방문별 score 변화 (line chart)
2. **변화율 차트**: 최근 방문 대비 변화율 (%) — 빨강=악화, 초록=호전, 회색=안정

**요약 카드**: "4회 방문 | 악화: 2 | 호전: 1 | 안정: 2"

**임상 활용**: 치료 시작 후 호전 여부, 질병 진행 속도 모니터링

---

### 6.7 UPDRS 추정

**목적**: 센서 데이터 기반 MDS-UPDRS Part III 항목별 score 추정

**시각화**: 수평 bar chart
- 각 UPDRS 항목 (3.17 Rest Tremor, 3.4 Finger Tapping, 3.5 Hand Movements, 3.7 Toe Tapping, 3.10 Gait 등)
- 추정 score (0~4) + confidence level
- 배경색: 초록(0~1), 노랑(1~2), 분홍(2~3), 빨강(3~4)
- UPDRS 기준선: 경미/경도/중등도 reference line

**근거 패널**: 각 항목의 추정 근거가 되는 sensor feature 값 표시

**중요**: 이 점수는 **참고용 estimate**이며, 최종 scoring은 임상 소견에 따릅니다.

---

### 6.8 Phase Portrait *(Novelty Visualization)*

**영감**: 물리학의 phase space 시각화

**목적**: movement amplitude vs. velocity를 2D 평면에 그려서 운동의 "건강함"을 한 장의 그림으로 판단

**해석**:
- **Healthy control**: 깨끗한 elliptical orbit (일정한 리듬과 진폭)
- **PD 환자**: 안쪽으로 수축하며 찌그러진 trajectory (진폭 감소 + 리듬 붕괴)
- **색상**: 파랑(시작) → 빨강(종료)으로 시간 경과 표시
- **START/END 마커**: 궤적의 시작점과 종료점

**임상 활용**: 기존의 1D 시계열 그래프로는 파악하기 어려운 amplitude-velocity 상호작용을 직관적으로 보여줌

---

### 6.9 Signature Wall *(Novelty Visualization)*

**영감**: NYT "Unemployment Lines" (2009) — 수백 개 지역의 실업률을 한 화면에 overlay

**목적**: 모든 환자의 movement waveform을 한 화면에 겹쳐서 "이 환자가 다른 환자들 사이에서 어디쯤인지" 즉시 비교

**시각화**:
- 배경: 모든 환자의 waveform (연한 회색, opacity 0.35)
- 선택 환자: 진단 그룹 색상으로 강조 (PD=빨강, Healthy=초록, DDx=노랑)
- Population ±1σ band: 전체 환자 평균 ± 표준편차 범위
- **Hover**: 마우스를 올리면 해당 환자의 ID와 group이 표시

**임상 활용**: 개별 환자의 movement pattern이 cohort 전체에서 얼마나 일탈하는지 직관적으로 파악

---

### 6.10 Evidence Map *(Novelty Visualization)*

**영감**: Information is Beautiful "Snake Oil" chart — 버블 크기/위치로 근거 수준 표시

**목적**: 어떤 movement feature에 가장 주목해야 하는지를 한 장의 버블맵으로

**시각화**:
- **X축**: Abnormality (Z-score vs healthy) — 비정상 정도
- **Y축**: Diagnostic Power — 해당 feature가 PD 진단에 얼마나 기여하는지
- **버블 크기**: 환자의 해당 feature 실제 severity
- **버블 색상**: Motor task 카테고리
- **사분면 라벨**: 오른쪽 위 = "주의 필요 (HIGH PRIORITY)", 왼쪽 아래 = "정상 범위 (LOW CONCERN)"

**임상 활용**: 30개(6 task × 5 feature) 데이터 포인트 중 어디에 임상적 주의를 집중해야 하는지 한눈에 파악

---

## 7. 핵심 Movement Feature 설명

| Feature | 설명 | 임상 의미 |
|---------|------|----------|
| **Tremor Power** | 가속도 신호의 4-6Hz 대역 파워 | 안정 시/자세 유지 시 떨림 강도 |
| **Amplitude Reduction** | 반복 동작에서 진폭이 점차 감소하는 정도 | Bradykinesia의 핵심 지표 (decrement) |
| **Rhythm Irregularity** | 반복 동작 간격의 변동계수 (CV) | 운동 리듬의 불규칙성 |
| **Left-Right Asymmetry** | 좌우 사지 간 movement feature 차이 | PD의 lateralized impairment |
| **Motion Instability** | 신호의 jerkiness (가속도 미분의 변동) | 움직임의 매끄러움 저하 |

---

## 8. Motor Task 설명

| Task | 설명 | 대응 UPDRS 항목 |
|------|------|----------------|
| **Finger Tapping** | 엄지와 검지를 빠르게 반복 두드리기 | 3.4 |
| **Hand Open/Close** | 손을 빠르게 반복 펴기/쥐기 | 3.5 |
| **Rest Tremor** | 양손을 무릎 위에 놓고 안정 시 떨림 관찰 | 3.17 |
| **Gait** | 일정 거리를 걷기 | 3.10 |
| **Toe Tapping** | 발가락을 빠르게 반복 두드리기 | 3.7 |
| **Touch Nose** | 손가락으로 코를 반복 터치 | 3.8 (proxy) |

---

## 9. 추후 확장 방향

### 실제 데이터 연결
1. **PADS 데이터**: JSON 메타데이터 → CSV 변환, 시계열 → FFT/peak detection으로 feature 추출
2. **TULIP 데이터**: 3명 전문가 rating 활용, MediaPipe로 영상 keypoint 추출

### 추가 기능
- 환자 간 비교 기능 (두 환자를 나란히 비교)
- PDF 리포트 자동 생성
- 실시간 센서 데이터 스트리밍 연결
- 약물 복용 전/후 비교 (ON/OFF state)

---

## 10. 발표용 요약

본 프로젝트는 파킨슨병 환자의 운동 기능을 **정량적으로 시각화**하여 의사의 **임상 의사결정을 지원**하는 Dash + Plotly 기반 full-stack interactive dashboard입니다.

스마트워치 센서 데이터에서 추출한 tremor, rhythm, amplitude, asymmetry 등의 movement feature를 기반으로:

1. 환자의 전반적 motor profile을 요약하고 (**환자 Overview**)
2. 어떤 task에서 어떤 이상이 있는지 정밀 분석하고 (**Task-Feature Heatmap**)
3. 좌우 비대칭을 정량 비교하고 (**좌우 비교**)
4. 반복 운동의 bradykinesia 지표를 계산하고 (**Rhythm & Amplitude**)
5. 건강대조군 대비 위치를 확인하고 (**Normative 비교**)
6. 방문별 경과를 추적하고 (**경과 추적**)
7. UPDRS 점수를 참고용으로 추정하고 (**UPDRS 추정**)
8. 물리학 기반 위상 공간 분석 (**Phase Portrait**)
9. 전체 cohort 대비 파형 비교 (**Signature Wall**)
10. 임상적 우선순위를 버블맵으로 제시합니다 (**Evidence Map**)

이를 통해 단순히 PD 여부를 예측하는 것이 아니라, **어떤 motor task에서 어떤 movement feature가 얼마나 비정상적인지**를 의사에게 정량적 근거로 제시합니다.
