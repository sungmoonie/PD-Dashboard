# Visual Consensus Dashboard for Parkinson’s Motor Assessment

파킨슨 운동평가의 의사 간 판단 차이를 줄이기 위한 시각적 합의 보조 대시보드

---

## 1. 프로젝트 개요

나는 데이터 시각화 수업 프로젝트를 진행 중이다.  
프로젝트는 파킨슨 운동평가에서 의사 간 판단 차이를 줄이기 위한 **full-stack visual analytics dashboard**이다.

### 프로젝트 제목

**Visual Consensus Dashboard for Parkinson’s Motor Assessment**  
파킨슨 운동평가의 의사 간 판단 차이를 줄이기 위한 시각적 합의 보조 대시보드

### 핵심 목표

이 프로젝트는 파킨슨 환자를 자동으로 진단하는 AI 시스템이 아니다.  
대신 의사가 환자의 운동 이상 패턴을 더 일관된 기준으로 해석할 수 있도록 돕는 시각화 대시보드이다.

### 문제의식

파킨슨 환자의 운동 기능 평가는 보통 짧은 과제 수행 영상, 임상 관찰, UPDRS 기반 점수, 센서 데이터 등을 바탕으로 이루어진다.

하지만 같은 영상을 보더라도 의사마다 주목하는 요소가 다를 수 있다.

예를 들어 어떤 의사는 떨림을 더 중요하게 보고, 어떤 의사는 움직임의 속도 저하, 좌우 비대칭, 반복 동작의 진폭 감소, 리듬 붕괴를 더 중요하게 볼 수 있다.

따라서 단순히 영상만 보여주는 방식은 **inter-rater variability**, 즉 의사 간 평가 차이를 줄이기 어렵다.

이 프로젝트는 스마트워치 센서 데이터와 영상 기반 움직임 특징을 공통된 시각적 기준으로 정렬해 보여줌으로써, **“어떤 과제에서, 어떤 움직임 특징이, 어느 정도로 비정상적인지”**를 의사가 함께 볼 수 있게 하는 visual analytics dashboard를 목표로 한다.

---

## 2. 사용 예정 데이터셋

## 2.1 PADS 데이터셋

PADS는 파킨슨 환자, 감별진단군, 건강대조군의 스마트워치 기반 신경학적 평가 데이터를 포함한다.

주요 특징:

- 양손 스마트워치의 acceleration / gyroscope 신호
- movement task 정보
- 인구통계
- 병력
- PD-specific non-motor symptoms
- 좌우 비대칭 분석에 적합

### 활용 방향

PADS는 스마트워치 기반 정량 motor profile을 만드는 데 사용한다.

특히 양손 스마트워치 데이터를 활용해 다음을 분석한다.

- 좌우 비대칭
- 떨림 강도
- 반복 동작의 리듬 불규칙성
- 움직임 크기 감소
- 움직임 불안정성

---

## 2.2 TULIP 데이터셋

TULIP은 파킨슨 환자 및 건강대조군의 임상 motor task 영상 데이터셋이다.

주요 특징:

- 6대 카메라로 촬영된 multi-view video
- 파킨슨 환자 및 건강대조군의 motor task 영상
- 3명의 clinical expert rating 포함
- multi-view video 기반 3D 움직임 복원과 임상 평가 비교에 적합

### 활용 방향

TULIP은 영상 기반 motion consensus view와 clinical disagreement view에 사용한다.

활용 예시:

- 영상 기반 keypoint trajectory
- motion trail prototype
- clinical expert rating 비교
- 의사 간 평가 차이가 큰 task 탐색

---

## 3. 초기 구현 전략

초기 prototype에서는 실제 데이터셋 전체를 바로 사용하지 않는다.

대신 demo CSV / JSON 데이터를 만들어 다음 기능을 먼저 구현한다.

- 환자 선택
- 환자별 overview 표시
- task-by-feature heatmap
- 좌우 비대칭 비교
- rhythm & amplitude time-series
- clinical rater disagreement
- motion trail prototype

즉, 먼저 **대시보드 구조와 인터랙션을 스케치**한 뒤 실제 데이터셋으로 확장한다.

---

## 4. 추천 기술 스택

Streamlit은 사용할 수 없고, Gradio는 꼭 쓸 필요가 없다.  
또한 full-stack 구현 느낌이 있어야 한다.

### 최종 추천 스택

| 영역 | 추천 도구 |
|---|---|
| Dashboard Framework | Dash by Plotly |
| Visualization | Plotly Express, Plotly Graph Objects |
| Backend | Dash callback 기반 Python backend |
| Server | Dash 내부 Flask server |
| Data Processing | pandas, numpy, scipy |
| Prototype Data Storage | CSV / JSON |
| Optional API Backend | FastAPI |
| Optional Database | SQLite 또는 PostgreSQL |
| Optional Video Processing | MediaPipe |

---

## 5. 왜 Dash를 사용하는가?

이 프로젝트에서는 React보다 Dash가 초기 구현에 더 적합하다.

### 이유

1. 프로젝트가 일반 웹앱보다 시각화 대시보드 중심이다.
2. Python 기반으로 pandas, numpy, scipy와 바로 연결할 수 있다.
3. Plotly 차트와 자연스럽게 연결된다.
4. Heatmap 클릭, dropdown 선택, 그래프 업데이트 같은 대시보드 상호작용을 callback으로 쉽게 구현할 수 있다.
5. Streamlit이 금지된 상황에서 Python 기반 interactive dashboard tool 중 가장 대중적이고 시각화 수업에 적합하다.
6. full-stack 구조를 명확하게 설명할 수 있다.

### Full-stack 구조 설명

#### Frontend layer

- Dash layout
- Dash components
- `dcc.Dropdown`
- `dcc.Tabs`
- `dcc.Graph`
- `html.Div`
- Plotly visualization

#### Backend layer

- Python callback functions
- patient filtering
- feature calculation
- asymmetry index calculation
- rhythm irregularity calculation
- clinical rater disagreement calculation

#### Data layer

- demo CSV / JSON files
- `patients.csv`
- `task_features.csv`
- `timeseries.csv`
- `ratings.csv`
- `motion_trail.json`

---

## 6. 전체 앱 구조

```text
visual-consensus-dashboard/
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── patients.csv
│   ├── task_features.csv
│   ├── timeseries.csv
│   ├── ratings.csv
│   └── motion_trail.json
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── figures.py
│   └── layout.py
│
└── assets/
    └── style.css
```

### 각 파일 역할

#### `app.py`

- Dash 앱 실행
- layout 불러오기
- callback 정의
- 사용자 입력과 그래프 업데이트 연결

#### `data_loader.py`

- CSV / JSON demo data 로드
- patient 목록 반환
- task 목록 반환
- feature matrix 반환

#### `feature_engineering.py`

- asymmetry index 계산
- rhythm irregularity 계산
- amplitude decrement 계산
- rater disagreement score 계산
- abnormality score 계산

#### `figures.py`

- Plotly figure 생성 함수 모음
- overview radar chart
- task-feature heatmap
- left-right mirror plot
- rhythm & amplitude time-series
- clinical disagreement chart
- motion trail plot

#### `layout.py`

- 전체 Dash layout 구성
- header, sidebar, tabs, cards, graphs 정의

#### `assets/style.css`

- 대시보드 디자인
- medical dashboard 느낌의 card layout, color, spacing

---

## 7. 대시보드 전체 화면 구성

### Header

- Dashboard title: **Visual Consensus Dashboard for Parkinson’s Motor Assessment**
- Subtitle: **Visual analytics for reducing inter-rater variability in motor assessment**
- Patient selector
- Optional task selector

### Main layout

- Left sidebar: patient selection and global filters
- Main content: tab-based dashboard
- Right panel or bottom panel: selected task / feature explanation

### Tabs

1. Patient Overview
2. Task-by-Feature Heatmap
3. Left-Right Mirror Plot
4. Rhythm & Amplitude Panel
5. Clinical Disagreement View
6. Motion Trail Prototype

---

# 8. Tab 1 — Patient Overview

## 목적

환자 한 명을 선택했을 때 전체 motor / non-motor profile을 빠르게 요약한다.

의사가 raw sensor signal이나 복잡한 영상을 보기 전에, 이 환자의 핵심 이상 패턴을 한눈에 파악하게 한다.

## UI 구성

### 상단

- Patient ID dropdown
- Diagnosis group badge: PD / Healthy Control / Differential Diagnosis
- Age
- Sex
- Handedness

### 요약 카드

- Overall Motor Abnormality Score
- Left-Right Asymmetry Score
- Tremor Intensity
- Motion Regularity
- Non-Motor Symptom Burden
- Top Abnormal Task

### 중앙 시각화

- Radar chart: tremor, amplitude reduction, rhythm irregularity, asymmetry, instability
- Bar chart: Top 3 abnormal tasks

### 하단 해석 문구

예시:

> This patient shows high rhythm irregularity and left-right asymmetry during finger tapping.

## 추천 차트

### 1. Radar chart

- Plotly Graph Objects `Scatterpolar`
- feature별 환자 profile을 한눈에 보여줌

### 2. Top abnormal tasks bar chart

- Plotly Express `bar`
- task별 abnormality score 순위 표시

### 3. Summary cards

- Dash `html.Div` cards

## 필요한 데이터

### `patients.csv`

```csv
patient_id,group,age,sex,handedness,motor_score,non_motor_score
P001,PD,68,F,right,72,41
```

### `task_features.csv`

```csv
patient_id,task,feature,value,normal_percentile
P001,finger_tapping,tremor_power,0.72,92
```

---

# 9. Tab 2 — Task-by-Feature Heatmap

## 목적

이 환자가 어떤 motor task에서 어떤 이상 패턴을 보이는지 보여준다.

의사에게 중요한 질문은 “전체적으로 나쁜가?”보다 **“어떤 동작에서 어떤 종류의 문제가 나타나는가?”**이다.

이 탭은 그 질문에 직접 답한다.

## 핵심 시각화

### Task-by-Feature Heatmap

#### Rows

- Rest tremor
- Finger tapping
- Hand opening-closing
- Toe tapping
- Gait
- Touch nose

#### Columns

- Tremor Power
- Movement Amplitude Reduction
- Rhythm Irregularity
- Left-Right Asymmetry
- Motion Instability / Jerkiness

#### Cell value

- normalized abnormality score, 0 to 1
- 또는 healthy control 대비 percentile

## 인터랙션

1. 환자 선택 시 heatmap 업데이트
2. heatmap cell 클릭 시:
   - selected task 저장
   - selected feature 저장
   - 하단 또는 오른쪽 detail panel 업데이트
   - 관련 Left-Right Mirror Plot 또는 Rhythm Panel로 연결

## 추천 차트

- Plotly Heatmap
- `plotly.express.imshow`
- 또는 `plotly.graph_objects.Heatmap`

### Dash clickData 사용

```python
Input("task-feature-heatmap", "clickData")
```

## 예시 해석

### Finger tapping × Rhythm Irregularity cell이 높을 때

> 반복 동작의 간격이 불규칙하여 의사 간 평가 차이가 발생할 가능성이 있음.

### Gait × Asymmetry cell이 높을 때

> 좌우 보행 패턴 차이가 커서 motor impairment 판단 근거가 될 수 있음.

---

# 10. Tab 3 — Left-Right Mirror Plot

## 목적

파킨슨 평가에서 중요한 좌우 비대칭을 직관적으로 보여준다.

좌우 움직임의 크기, 떨림, 리듬 불규칙성, 속도 차이를 같은 기준에서 비교한다.

## 핵심 지표

```text
Asymmetry Index = |Left - Right| / ((Left + Right) / 2)
```

### 값이 낮으면

- 좌우 움직임이 유사함

### 값이 높으면

- 한쪽 움직임이 더 제한되거나 불안정함

## UI 구성

### 상단

- Patient ID
- Task selector
- Asymmetry Index summary card
- Normal percentile comparison

### 중앙

- Mirror bar chart
- 왼쪽 feature 값은 음수 방향
- 오른쪽 feature 값은 양수 방향

### Feature rows

- Amplitude
- Tremor Power
- Rhythm Irregularity
- Velocity
- Instability

## 추천 차트

- Plotly Graph Objects `Bar`
- `orientation="h"`
- left values as negative
- right values as positive

## 예시 시각화 개념

```text
Left side                      Right side
Amplitude       ███████ | ████
Tremor          ███     | ███████
Rhythm          █████   | ██████
Velocity        ██████  | ███
```

## 필요한 데이터

### `left_right_features.csv`

```csv
patient_id,task,feature,left_value,right_value,asymmetry_index,normal_percentile
P001,finger_tapping,amplitude,0.42,0.71,0.51,91
```

---

# 11. Tab 4 — Rhythm & Amplitude Panel

## 목적

반복 운동에서 의사마다 다르게 해석할 수 있는 **“리듬 불규칙성”**과 **“점점 작아지는 움직임”**을 정량화한다.

“불규칙해 보인다”라는 주관적 판단을 tap interval, amplitude decay, rhythm variability로 보여준다.

## 추천 task

초기 prototype에서는 하나만 집중해도 충분하다.

### 추천

- Finger tapping

### 추가 가능

- Hand opening-closing
- Toe tapping
- Touch nose

## UI 구성

### 상단

- Patient ID
- Task selector
- Rhythm Irregularity Score
- Amplitude Decrement Score

### 중앙 왼쪽

- Time-series line chart
- left amplitude / right amplitude over time

### 중앙 오른쪽

- Detected peaks
- tap interval chart
- rhythm variability

### 하단

- amplitude decay chart
- early repetition average vs late repetition average

## 추천 차트

### 1. Time-series line chart

- `plotly.express.line`
- x = time
- y = left_amplitude, right_amplitude

### 2. Peak marker overlay

- `plotly.graph_objects.Scatter` markers

### 3. Tap interval bar chart

- x = repetition number
- y = interval duration

### 4. Amplitude decrement line chart

- x = repetition number
- y = peak amplitude

## 계산 지표

### tap_interval

- detected peak 간 시간 차이

### rhythm_irregularity

- standard deviation of tap intervals
- 또는 coefficient of variation

```text
coefficient of variation = std(intervals) / mean(intervals)
```

### amplitude_decrement

- early peak amplitude average와 late peak amplitude average의 차이

```text
amplitude_decrement = (mean(first_3_peaks) - mean(last_3_peaks)) / mean(first_3_peaks)
```

## 핵심 메시지

이 탭은 **“의사 간 판단 차이를 줄인다”**는 프로젝트 주제와 직접 연결된다.

의사가 주관적으로 “리듬이 안 좋다”, “점점 작아진다”고 판단하는 부분을 수치와 그래프로 보여준다.

---

# 12. Tab 5 — Clinical Disagreement View

## 목적

프로젝트 제목이 **Visual Consensus Dashboard**이므로 이 탭은 매우 중요하다.

3명의 clinical expert rating이 서로 다른 경우, 그 차이가 어떤 movement feature와 관련 있는지 보여준다.

## UI 구성

### 상단

- Patient ID
- Task selector
- Disagreement score
- Rating range
- Rating variance

### 중앙

- Rater score comparison bar chart

예시:

```text
Rater 1: 3
Rater 2: 2
Rater 3: 3
```

### 오른쪽

- Rater confidence comparison
- confidence bar or dot plot

### 하단

- Evidence feature panel
- selected task의 tremor, amplitude, rhythm, asymmetry feature 표시
- “왜 평가가 갈릴 수 있는지” 설명

## 추천 차트

### 1. Grouped bar chart

- rater별 score 비교

### 2. Dot plot

- rater별 confidence 표시

### 3. Evidence mini heatmap

- 해당 task의 feature severity 표시

### 4. Consensus summary card

- consensus score
- disagreement level: low / moderate / high

## 필요한 데이터

### `ratings.csv`

```csv
patient_id,task,rater_id,score,confidence
P001,finger_tapping,R1,3,0.81
```

### `task_features.csv`

```csv
patient_id,task,feature,value,normal_percentile
P001,finger_tapping,rhythm_irregularity,0.81,95
```

## 계산 지표

```text
rating_range = max(score) - min(score)
rating_variance = variance(scores)
consensus_score = mean(scores)
```

### disagreement_level

- range 0: low
- range 1: moderate
- range >= 2: high

## 예시 해석

> Rater disagreement is moderate. Rater 2 assigned a lower score than Rater 1 and Rater 3. The movement evidence shows high rhythm irregularity but only moderate amplitude reduction, which may explain the disagreement.

---

# 13. Tab 6 — Motion Trail Prototype

## 목적

영상 기반 움직임의 시간성을 시각적으로 보여준다.

Stick figure만 보여주는 것보다 손목, 손끝, 발목 등의 궤적을 시간 흐름에 따라 보여주는 것이 더 직관적이다.

## 구현 난이도 조절

### 초기 prototype

- 실제 영상 처리 없이 demo keypoint JSON을 사용한다.
- 손목 또는 손끝 좌표를 시간 순서대로 연결한다.
- Plotly scatter/line 또는 SVG로 motion trail을 그린다.

### 확장 버전

- TULIP 영상에서 MediaPipe Pose 또는 MediaPipe Hands로 keypoint 추출
- backend에서 좌표 JSON 생성
- frontend에서 video overlay 또는 trajectory plot 표시

## UI 구성

### 상단

- Patient ID
- Task selector
- Camera view selector, optional

### 중앙

- Motion trail plot
- x coordinate vs y coordinate
- line color 또는 marker size로 time 또는 velocity 표현

### 오른쪽

- velocity over time
- trajectory length
- movement instability score

### 하단

- optional video placeholder

## 추천 차트

### 1. Plotly scatter line

- x = keypoint_x
- y = keypoint_y
- mode = `lines+markers`

### 2. Velocity-colored trajectory

- marker color = velocity
- 또는 line segments by time

### 3. Optional frame preview

- static image placeholder

## 필요한 데이터

### `motion_trail.json`

```json
{
  "patient_id": "P001",
  "task": "finger_tapping",
  "keypoint": "right_index_finger",
  "frames": [
    {"frame": 1, "time": 0.00, "x": 0.42, "y": 0.61, "velocity": 0.12},
    {"frame": 2, "time": 0.03, "x": 0.43, "y": 0.60, "velocity": 0.15}
  ]
}
```

---

# 14. Demo 데이터 설계

## `patients.csv`

```csv
patient_id,group,age,sex,handedness,motor_score,non_motor_score
P001,PD,68,F,right,72,41
P002,PD,61,M,right,58,35
P003,Healthy,59,F,left,12,8
P004,Differential,65,M,right,43,22
P005,PD,72,M,left,80,46
```

## `task_features.csv`

```csv
patient_id,task,feature,value,normal_percentile
P001,finger_tapping,tremor_power,0.72,92
P001,finger_tapping,movement_amplitude_reduction,0.69,88
P001,finger_tapping,rhythm_irregularity,0.81,95
P001,finger_tapping,left_right_asymmetry,0.64,90
P001,finger_tapping,motion_instability,0.58,84
P001,gait,tremor_power,0.22,55
P001,gait,movement_amplitude_reduction,0.48,78
P001,gait,rhythm_irregularity,0.52,81
P001,gait,left_right_asymmetry,0.71,93
P001,gait,motion_instability,0.67,89
```

## `left_right_features.csv`

```csv
patient_id,task,feature,left_value,right_value,asymmetry_index,normal_percentile
P001,finger_tapping,amplitude,0.42,0.71,0.51,91
P001,finger_tapping,tremor_power,0.77,0.38,0.68,94
P001,finger_tapping,rhythm_irregularity,0.82,0.56,0.38,86
P001,finger_tapping,velocity,0.49,0.68,0.32,80
```

## `timeseries.csv`

```csv
patient_id,task,time,left_amplitude,right_amplitude,left_velocity,right_velocity
P001,finger_tapping,0.00,0.81,0.66,1.20,1.05
P001,finger_tapping,0.05,0.79,0.62,1.18,1.01
P001,finger_tapping,0.10,0.72,0.59,1.10,0.98
P001,finger_tapping,0.15,0.69,0.57,1.05,0.95
```

## `ratings.csv`

```csv
patient_id,task,rater_id,score,confidence
P001,finger_tapping,R1,3,0.81
P001,finger_tapping,R2,2,0.64
P001,finger_tapping,R3,3,0.77
P001,gait,R1,2,0.71
P001,gait,R2,3,0.69
P001,gait,R3,2,0.74
```

---

# 15. 핵심 Callback 설계

## Callback 1 — Patient Overview 업데이트

### Input

- `patient-dropdown.value`

### Output

- overview cards
- radar chart
- top abnormal tasks chart

---

## Callback 2 — Heatmap 업데이트

### Input

- `patient-dropdown.value`

### Output

- task-feature heatmap

---

## Callback 3 — Heatmap click interaction

### Input

- `task-feature-heatmap.clickData`
- `patient-dropdown.value`

### Output

- selected task store
- selected feature store
- detail explanation panel

---

## Callback 4 — Left-Right Mirror Plot 업데이트

### Input

- `patient-dropdown.value`
- `task-dropdown.value`

### Output

- left-right mirror plot
- asymmetry summary card

---

## Callback 5 — Rhythm & Amplitude Panel 업데이트

### Input

- `patient-dropdown.value`
- `task-dropdown.value`

### Output

- rhythm amplitude time-series
- tap interval chart
- amplitude decrement chart

---

## Callback 6 — Clinical Disagreement View 업데이트

### Input

- `patient-dropdown.value`
- `task-dropdown.value`

### Output

- clinical disagreement chart
- evidence feature panel

---

## Callback 7 — Motion Trail 업데이트

### Input

- `patient-dropdown.value`
- `task-dropdown.value`

### Output

- motion trail plot

---

# 16. 추천 Plotly 차트 함수

`figures.py`에 다음 함수들을 만든다.

## 1. `make_overview_radar(patient_features)`

- Scatterpolar 기반 radar chart

## 2. `make_top_tasks_bar(task_scores)`

- `px.bar` 기반 abnormal task ranking

## 3. `make_task_feature_heatmap(matrix_df)`

- `px.imshow` 또는 `go.Heatmap`

## 4. `make_left_right_mirror_plot(left_right_df)`

- `go.Bar` horizontal
- left values negative
- right values positive

## 5. `make_rhythm_timeseries(timeseries_df)`

- `go.Scatter` line chart
- left / right amplitude over time

## 6. `make_tap_interval_chart(interval_df)`

- `px.bar`

## 7. `make_amplitude_decrement_chart(peaks_df)`

- `px.line`

## 8. `make_rater_disagreement_chart(ratings_df)`

- `px.bar` or `go.Bar`

## 9. `make_motion_trail_plot(motion_df)`

- `go.Scatter` x/y trajectory

---

# 17. 디자인 방향

Medical analytics dashboard 느낌으로 디자인한다.

## 스타일

- clean
- clinical
- high readability
- card-based layout
- light background
- navy / blue / gray 중심
- abnormal value는 red / orange 계열로 강조
- normal / stable value는 blue / green 계열로 표현
- 너무 화려한 디자인보다 의사가 보기 쉬운 dashboard 느낌

## Layout

- Header with title
- Sidebar for filters
- Main tab content
- Cards with rounded corners and subtle shadows
- Plotly charts inside cards

---

# 18. 프로젝트에서 강조해야 할 분석 포인트

1. 단순 진단 예측이 아니라 **visual consensus support**이다.
2. 의사가 같은 영상을 같은 기준으로 해석하도록 돕는다.
3. 원본 영상이나 skeleton만 보여주는 것이 아니라, 임상적으로 의미 있는 movement features로 정렬한다.
4. task-by-feature heatmap을 통해 어떤 과제에서 어떤 문제가 나타나는지 보여준다.
5. left-right mirror plot을 통해 파킨슨 평가에서 중요한 좌우 비대칭을 보여준다.
6. rhythm & amplitude panel을 통해 반복 운동의 리듬 붕괴와 진폭 감소를 정량화한다.
7. clinical disagreement view를 통해 의사 간 평가 차이와 그 원인을 movement evidence와 연결한다.
8. motion trail prototype은 영상 기반 움직임의 시간성을 시각적으로 보여주는 확장 기능이다.

---

# 19. Claude에게 요청할 최종 산출물

위 내용을 바탕으로 **Dash + Plotly 기반 full-stack 시각화 대시보드 prototype 코드**를 설계해줘.

원하는 산출물:

1. 전체 프로젝트 폴더 구조
2. `app.py` 코드
3. `src/data_loader.py` 코드
4. `src/feature_engineering.py` 코드
5. `src/figures.py` 코드
6. `src/layout.py` 코드
7. `assets/style.css` 코드
8. demo CSV / JSON 데이터 생성 코드
9. `requirements.txt`
10. `README.md` 실행 방법

## 중요 조건

- Streamlit은 사용하지 말 것
- Gradio도 사용하지 말 것
- Dash + Plotly 중심으로 구현할 것
- 실제 데이터셋 없이도 실행 가능한 demo data를 포함할 것
- 환자 선택, task 선택, heatmap click interaction이 작동해야 함
- 코드가 너무 복잡하지 않게, 수업 프로젝트용 prototype 수준으로 작성할 것

---

# 20. 팀 최종 추천안

| 항목 | 최종 추천 |
|---|---|
| 메인 도구 | Dash |
| 시각화 | Plotly Express + Plotly Graph Objects |
| 데이터 처리 | pandas, numpy, scipy |
| 백엔드 | Dash callback 기반 Python backend |
| 데이터 저장 | demo CSV / JSON |
| Optional backend | FastAPI |
| Optional DB | SQLite / PostgreSQL |
| 금지 | Streamlit |
| 굳이 안 써도 됨 | Gradio |
| MVP 필수 탭 | Overview, Heatmap, Mirror Plot, Rhythm Panel, Clinical Disagreement |
| 선택 탭 | Motion Trail Prototype |

---

# 21. 발표용 한 문단 설명

본 프로젝트는 파킨슨 운동평가에서 의사 간 판단 차이가 발생하는 문제를 줄이기 위한 **Dash + Plotly 기반 full-stack visual analytics dashboard**이다.

Frontend는 Dash layout과 Plotly interactive graph components로 구성하고, backend는 Python callback, pandas 기반 feature processing, demo dataset loading으로 구현한다.

대시보드는 Patient Overview, Task-by-Feature Heatmap, Left-Right Mirror Plot, Rhythm & Amplitude Panel, Clinical Disagreement View, Motion Trail Prototype으로 구성된다.

이를 통해 단순히 PD 여부를 예측하는 것이 아니라, 어떤 motor task에서 어떤 움직임 특징이 비정상적으로 나타나는지, 그리고 그 특징이 의사 간 평가 차이와 어떻게 연결되는지를 시각적으로 설명한다.
