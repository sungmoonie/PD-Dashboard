# Config Guide

`pads_matched/configs`는 데이터 다운로드/매핑에 필요한 설정 파일을 관리합니다.

## Files

### `tulip_ds.json`

TULIP 비디오 다운로드 설정 파일입니다.

- `tulip_video_folder_link`
  - subject 단위(`TULIP_001` 등)로 task별 Google Drive **폴더 링크**를 정의합니다.
- `metadata.tulip_pads_matched_base_dir`
  - 다운로드 결과를 저장할 기준 디렉터리입니다.
- `metadata.tulip_gdrive_link`
  - 전체 TULIP 원본 Google Drive 루트 링크입니다.

## Used By

- `pads_matched/scripts/download_tulip_video.sh`
  - 위 설정을 읽어 `by_tulip/TULIP_XXX/videos/{task_name}/` 구조로 자동 다운로드합니다.

## Path Convention

예시 저장 구조:

```text
pads_matched/by_tulip/
  TULIP_001/
    videos/
      17. Toe_tapping_left/
      18. Toe_tapping_right/
      26. Resting & hand tremor/
```