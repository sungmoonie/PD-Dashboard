# TULIP Video Download Guide

`download_tulip_video.sh`는 `configs/tulip_ds.json`에 정의된 Google Drive 폴더 링크를 읽어
TULIP 영상을 환자/태스크별로 자동 다운로드합니다.

## Prerequisite

```bash
pip install gdown
```

## Run

`PD-Dashboard` 루트에서 실행:

```bash
bash pads_matched/scripts/download_tulip_video.sh
```

## Config

- 설정 파일: `pads_matched/configs/tulip_ds.json`
- 사용 키:
  - `tulip_video_folder_link`: `TULIP_XXX`별 task 폴더 링크
  - `metadata.tulip_pads_matched_base_dir`: 다운로드 기준 디렉터리

## Output Structure

다운로드 결과는 아래 구조로 저장됩니다.

```text
pads_matched/by_tulip/
  TULIP_{index}/
    videos/
      {task_name}/
        Camera1.mp4
        Camera2.mp4
        Camera3.mp4
        Camera4.mp4
        Camera5.mp4
        Camera6.mp4
```

예시:
`pads_matched/by_tulip/TULIP_001/videos/17. Toe_tapping_left/Camera1.mp4`