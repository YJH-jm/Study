# SMPL(Skinned Multi-Person Linear model)
학계에서 가장 standard하게 사용햐는 body model <br>
후에 더 좋은 body model이 나왔지만 이미 SMPL을 이용한 논문이 많아서 이미 conventional하게 자리 잡음

<br>
<br>

## Background
논문을 더 쉽게 이해하기 위한 기본 지식

### Shape parameter

<br>

<img src="./images/3/1.png" width=40%>

<br>


- 10개의 실수값 (PCA coefficients)로 구성된 shape vector
- 각 실수값은 신장(tall/short) 축과 같은 특정 축 방향에서 object의 팽창, 수축의 정도로 해석 가능

<br>
<br>

### Pose parameter

<br>

<img src="./images/3/2.png" width=40%>

<br>

- 24x3 실수값으로 구성된 pose vector로, 각 joint parameter에 대응하는 relative rotation을 보존 
- 각 rotation은 axis-angle rotation representation에서 임의의 3차원 벡터로 인코딩 

<br>
<br>

### Model output
 - 24개의 관절에 대해 각각 3차원 실수값이 부여되지만, 이는 axis-angle ratation representation으로 인코딩 된 형태
 - 부모 관절에 대한 상댜적 회전을 의미
 - 24개의 계층은 pelvis에서부터 시작하고 이 pelvis의 관절 위치는 roo / cam_trans에 의해 상대적으로 정의
 - pelvis에 대해 자식 관계인 나머지 23개의 관절이 상대적으로 정의

 <br>
 <br>


## 3. Model Formulation
- SCAPE와 같이 identity-dependent shape와 non-rigid pose dependent shape
- SCAPE와 다르게 corrective blend shape를 사용한 vertex 기반의 skinning approach
- **Single blend shape** 는 vertex offset들을 합친 벡터를 의미
- Artist가 만든 mesh는 **Vertex=6890(N)** , **Joint=23(K)** 로 구성

<br>

<p align=center><img src="./images/3/3.png" width=20%></p>

<br>

- $\bar{T}$ 
    - 평균 template shape은 zero pose ($ \vec{\theta^{*}}$)에서의 N개의 vertex의 concat
        - 즉, $\beta$ 에서 나온 평균 체형을 가진 사람의 T-pose를 의미
    - 색은 Skinning weight를 visualize 한 것
    - $\bar{T}\in \mathbb{R}^{3N}$ 
- $W$
    - **Blend weight의 모음**을 의미하는데, Skinning weight를 의미하는 듯
        - Skinning weight : 각 vector가 어떤 joint에 가장 많은 영향을 받는지
    - $W\in \mathbb{R}^{N\times K}$

<br>

<p align=center><img src="./images/3/4.png" width=20%></p>

<br>

- **$B_{s}(\vec{\beta })$**
    - Blend shape function
    - $\beta$ 의 coefficient를 얻는 함수 
    - 입력으로 shape parameter ($\vec{\beta}$), 출력으로는 subject identity의 shape를 조각한 결과
    - $\mathbb{R}^{\left| \vec{\beta}\right|}\mapsto \mathbb{R}^{3N}$
    - 평균 shape에 체형에 해당하는 Blend shape function을 더해줌
- **$J(\vec{\beta})$**
    - K개의 joint의 위치를 예측하는 함수
    - (b)의 흰색 점들 의미
    -  $\mathbb{R}^{\left| \vec{\beta}\right|}\mapsto \mathbb{R}^{3K}$

<br>

<p align=center><img src="./images/3/5.png" width=20%></p>

<br>

- **$B_{p}(\vec{\theta})$**
    - Pose-dependent blend shape function 
    - pose parameter ($\theta$)를 input으로 pose-dependent 변형의 효과를 설명
        - 즉, pose dependent corrective를 설명함
    - 이 함수의 corrective blend shape은 T-pose에 더해짐
    - 
    - $\mathbb{R}^{\left| \vec{\theta}\right|}\mapsto \mathbb{R}^{3N}$

<br>

<p align=center><img src="./images/3/6.png" width=20%></p>

<br>

- $W$
    - Standard blend skinning function
    - 추츨한 관절의 중심부 주변의 vertex를 회전하는데 blend weight으로 smoothing 하여 진행
- $M(\vec{\beta}, \vec{\theta}, \Phi)$
    - 결과로 생성된 모델
    - shape, pose parameter들을 vertex들로 맵핑시켜줌
    - $\Phi$ 는 학습된 model parameter 의미

<br>

- LBS와 DQBS 모두 사용 
- Skinning method



<br>

**Blend skinning**
- 몸의 pose는 골격 구조로 ($\vec{\omega_{k}}\in\mathbb{R}^{3}$) 정의
- Kinematic tree 구조에서 부모 관절들에 대한 k 의 상대적인 회전에 대한 축-각도 표현(axis-angle representation)
- 골격 장치는 K=23개의 관절을 가지고 있기 때문에 pose parameter 인 $\vec{\theta}$ 는 $[\vec{\omega}^{T}_{0},... ,\vec{\omega}^{T}_{K}]^{T}$ 으로 표현
- $|\vec{\theta}|$ = 3 x 23 + 3 = 72개의 parameter들
    - 3가지 요소가 추가로 존재하는데 이는 root 방향
- $\bar{\omega}=\frac{\vec{\omega}}{\left\|\vec{\omega} \right\|}$ 는 단위 크기 회전 축을 의미
- 모든 관절 j에 대한 축 각도는 회전 행렬로 변환
    - Rodrigues formula를 통해

    $$exp(\vec{\omega}_{j})=I+\hat{\bar{\omega}}_{j}\sin{||\vec{\omega}_{j}||}+\hat{\bar{\omega}}_{j}^{2}\cos {||\vec{\omega}_{j}||}$$

    - $I$ 는 3x3 단위 행랼
    - $\hat{\bar{\omega}}$ 는 $\bar{\omega}$의 대칭 행렬  

<br>
<br>


## [code](https://github.com/vchoutas/smplx/tree/main)
[참고](https://khanhha.github.io/posts/SMPL-model-introduction/) 
### [body_models.py](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py)
```python
class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        **kwargs
    ) -> None:
        ''' SMPL model constructor
            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape coefficients.\n'
                  f'num_betas={num_betas}, shapedirs.shape={shapedirs.shape}, '
                  f'self.SHAPE_SPACE_DIM={self.SHAPE_SPACE_DIM}')
            num_betas = min(num_betas, shapedirs.shape[-1])
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age == 'kid':
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(
                v_template_smil - data_struct.v_template, axis=2)
            shapedirs = np.concatenate(
                (shapedirs[:, :, :num_betas], v_template_diff), axis=2)
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros(
                    [batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return 'SMPL'

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f'Gender: {self.gender.upper()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
        ]
        return '\n'.join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None, # 3D 길이 및 체형을 나타내는 파라미터
        body_pose: Optional[Tensor] = None, # theta에서 root joint를 제외한 나머지 joint의 pose
        global_orient: Optional[Tensor] = None, # theta에서 root joint의 pose
        transl: Optional[Tensor] = None, # 3D translation 3차원 vector, 최종 output mesh의 global 위치
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model
            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)
            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        # lbss function에서 vertex와 joint를 얻고 
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output

```

<br>
<br>


### [lbs.py](https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/smplx/lbs.py)
```python
def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor, # Principal Components들 
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    ## v-template : T-Pose를 한 average 체형
    ## shapedirs : Principal components
    ## T-pose에 beta만 고려(체형만)한 결과
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    ## v_posed도 아직 pose를 취하기 전
    ## Pose dependent correctives 를 t-pose space에서 구현
    v_posed = pose_offsets + v_shaped


    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    ## lbs_weights = skinning weight
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)

    # lbs equation을 pytorch로 구현한 것 
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed 
```