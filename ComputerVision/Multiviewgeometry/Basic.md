#
## Local Featrue
### Local Image Features
- 이미지의 interesting 한 부분
- 많은 computer vision 알고리즘의 시작

<br>
<br>

### Local Image Feature 들의 활용
- Image representation
    - local feature들을 모아 image-level descriptor 생성
    - Object appearance modeling
        - Pose 변화에 민감하지 않고, 부분적인 occlusion에 강함
- Multiple views 사이의 matching을 찾음
    - Stereo matching
    - 3D reconstruction
    - Image stitching

<br>
<br>

### Good Local Featue
- Texture less 한 영역이나 line만 존재하는 영역보다 corner나 boundary 영역이 명확한 부분이 더 좋음
- 좋은 local visual feature가 되기 위한 조건
    1. Saliency
        - Feature는 이미지의 interesting 한 부분이 포함되어야 함
    2. Locality
        - 

<br>
<br>


## Model Fitting