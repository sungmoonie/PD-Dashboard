## MeTRAbs Joint / Feature Extraction Script

`extract_metrabs_joint_features.py`는 폴더 단위 비디오(`*.mp4`)를 입력으로 받아:

- `Camera{idx}_joint.csv` (프레임별 joint 좌표)
- `Camera{idx}_features.csv` (joint 기반 프레임 feature)

를 생성합니다.

task별로 feature 컬럼이 달라집니다.
- `Toe_tapping_*`: 발/발목/무릎 중심 (`left_toe_speed`, `toe_tapping_rate_proxy`, `toe_lr_asymmetry` 등)
- `Resting & hand tremor`: 어깨~손 중심 (`left_wrist_speed`, `left_tremor_amp_proxy`, `upper_limb_lr_asymmetry` 등)
- 그 외 task: 공통 generic feature(`mean_joint_speed` 등)

기본 체크포인트 경로는 `src/tulip_source/metrabs/ckpt/metrabs_eff2l_384px_800k_28ds_pytorch` 입니다.

### 실행 예시

```bash
cd PD-Dashboard/visual-consensus-dashboard/src/tulip_source

python3 scripts/extract_metrabs_joint_features.py \
  --video_dir "PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos/17. Toe_tapping_left" \
  --save_dir "PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos_feature"
```

### 출력 파일

`--save_dir/{task_name}` 아래에 아래 형식으로 저장됩니다.  
(`task_name`은 `--video_dir`의 마지막 폴더명입니다. 예: `17. Toe_tapping_left`)

- `PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos_feature/17. Toe_tapping_left/Camera1_joint.csv`
- `PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos_feature/17. Toe_tapping_left/Camera1_features.csv`
- ...
- `PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos_feature/17. Toe_tapping_left/Camera6_joint.csv`
- `PD-Dashboard/pads_matched/by_tulip/TULIP_001/videos_feature/17. Toe_tapping_left/Camera6_features.csv`

### 옵션

- `--model_dir`: MeTRAbs 체크포인트 경로 (기본값: 코드 내 `DEFAULT_MODEL_DIR`)
- `--skeleton`: skeleton 이름 (기본 `smpl_24`)
- `--device`: `auto|cuda|cpu`
- `--batch_size`: 배치 크기 (기본 8)
- `--num_aug`: 추론 augmentation 횟수 (기본 5)
