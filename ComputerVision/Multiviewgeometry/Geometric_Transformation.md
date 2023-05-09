# 2D Geometric Transformation 
카메라 모델 이해하여 3D의 실제 생활 좌표를 2D 이미지 좌표로 맵핑 시키기 위함

<br>
<br>

## 2D Geometric Primitive 
### 2D Points
- 이미지 픽셀의 좌표
- Inhomogeneous coordinate 상에서는 
    - 일반적인 coordinate 시스템을 사용
    - $\bold{x}=[x,y]^{T}\in\mathbb{R}^{2}$
- Homogeneous coordinate 상에서는
    - CV에서 일반적으로 사용
    