# Video to Pose3D

> Predict 3d human pose from video

<p align="center"><img src="asset/kunkun_alphapose.gif" width="100%" alt="" /></p>

## Prerequisite

1. Environment
   - Linux system
   - Python > 3.6 distribution
2. Dependencies
   - **Packages**
      - Pytorch > 1.0.0
      - [torchsample](https://github.com/MVIG-SJTU/AlphaPose/issues/71#issuecomment-398616495)
      - [ffmpeg](https://ffmpeg.org/download.html)
      - tqdm
      - pillow
      - scipy
      - pandas
      - h5py
      - visdom
      - nibabel
      - opencv-python (install with pip)
      - matplotlib

## Dataset

우리 모델은 [Human3.6M](http://vision.imar.ro/human3.6m) 와 [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) dataset들을 사용해서 평가됩니다.

### Human3.6M

우리는 [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)과 같은 방법으로 Human3.6M 데이터셋을 설정했습니다.  전처리된 데이터는 [여기](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing)에서 다운로드할 수 있습니다.  `data_2d_h36m_gt.npz` 는 2D  keypoint의 ground truth입니다. `data_2d_h36m_cpn_ft_h36m_dbb.npz` 는 [CPN](https://github.com/GengDavid/pytorch-cpn)에 의해 얻어진 2D keypoint 입니다. `data_3d_h36m.npz` 는 3D human joints(관절?)의 ground truth입니다. 이들 모두 `./dataset` 디렉토리에 위치시켜야 합니다.

### MPI-INF-3DHP

우리는 스스로 MPI-INF-3DHP dataset을 설정했습니다. 우리는 `data_to_npz_3dhp.py` 와`data_to_npz_3dhp_test.py`를 사용해서 오리지널 데이터의 `.mat` 포맷 파일들을 전처리해 `.npz` 포맷으로 변환했습니다. 전처리된 데이터는 [여기](https://drive.google.com/file/d/11eBe175Rgj6IYrwZwa1oXTOyHPxGuWyi/view?usp=sharing)에서 다운로드할 수 있습니다. 이들을 모두 `./dataset` 디렉토리에 위치시키십시오. 추가적으로 만약 당신이 PCK(Percentage of Correct keypoints)와 AUC(???) 메트릭을 해당 dataset에 추가하고 싶다면, 당신은 [해당 공식 웹사이트](https://vcai.mpi-inf.mpg.de/3dhp-dataset/)에서 original dataset 또한 다운로드해야 합니다. Dataset을 다운로드 한 후에는, 해당 레포지토리의 `./3dhp_test` 폴더 아래에 test set 안에 TS1부터 TS6 폴더를 모두 위치시키십시오.

## Pre-trained 모델 평가하기

당신은 pre-trained model을 [여기서](https://drive.google.com/file/d/1vLtC86_hs01JKKRQ6akvdH5QDKxt71cY/view?usp=sharing) 다운로드할 수 있습니다. 이들을 `./checkpoint` 디렉토리 내부에 위치시키시면 됩니다. 

### Human 3.6M

the ground truth of 2D keypoints에 대해 우리의 P-STMO-S model을 평가하고 싶다면 아래 명령어를 입력:

```bash
python run.py -k gt -f 243 -tds 2 --reload 1 --previous_dir
checkpoint/PSTMOS_no_refine_15_2936_h36m_gt.pth
```

아래 모델들은 **CPN**을 통해 얻은 2D keypoint를 입력으로 사용해 훈련됩니다.

우리의 P-STMO-S model를 평가하기 위해서는 다음 명령어를 입력 :

```bash
python run.py -f 243 -tds 2 --reload 1 --previous_dir checkpoint/PSTMOS_no_refine_28_4306_h36m_cpn.pth
```

우리의 P-STMO model을 평가하기 위해서는 다음 명령어를 입력 :

```bash
python run.py -f 243 -tds 2 --reload 1 --layers 4 --previous_dir checkpoint/PSTMO_no_refine_11_4288_h36m_cpn.pth
```

[ST-GCN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Exploiting_Spatial-Temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_ICCV_2019_paper.pdf)에서 제안됐던 개선된 모듈을 사용해 우리의 P-STMO model을 평가하려면 다음 명령어를 입력:

```bash
python run.py -f 243 -tds 2 --reload 1 --refine_reload 1 --refine --layers 4 --previous_dir checkpoint/PSTMO_no_refine_6_4215_h36m_cpn.pth --previous_refine_name checkpoint/PSTMO_refine_6_4215_h36m_cpn.pth
```

### MPI-INF-3DHP

MPI-INF-3DHP dataset에 대해서 우리의 P-STMO-S model을 평가하려면 다음 명령어를 입력:

```bash
python run_3dhp.py -f 81 --reload 1 --previous_dir checkpoint/PSTMOS_no_refine_50_3203_3dhp.pth
```

이후 3D 포즈 예측은 `./checkpoint/inference_data.mat`으로 저장됩니다. 이 결과는 `./3dhp_test/test_util/mpii_test_predictions_py.m`명령어를 실행해 Mattlab을 사용함으로써 평가할 수 있습니다. 최종 평가 결과는 프레임 수에 대한 평균 sequencewise(순차적) 평가 결과를 평균화해서 얻은 `./3dhp_test/mpii_3dhp_evaluation_sequencewise.xlsx`을 통해 확인할 수 있습니다. 시각화를 위해서 당신은 `./common/draw_3d_keypoint_3dhp.py` 와 `./common/draw_2d_keypoint_3dhp.py`를 사용할 수 있습니다.

처음부터 교육파트 생략

## Testing on in-the-wild videos

Custom video들에서 우리의 모델을 평가하려면, 당신은 [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)와 같은 off-the-shelf 2D keypoint detector를 사용해 이미지에서 2D 포즈를 생성하고, 우리의 모델을 사용해 3D 포즈를 만들 수 있습니다. 2D keypoint detector들은 Human3.6M과는 다른 방식으로 인간의 joint 순서를 정의하는 COCO dataset을 사용해서 훈련됩니다. **따라서 기존의 detector와 호환되도록 모델을 다시 학습해야 합니다.** 우리의 모델은 [여기서](https://drive.google.com/file/d/1xO0Oo1yV3-5eQSIBUyIzelvaAwWpLqM5/view?usp=sharing) 다운로드 할 수 있는 COCO format의 2D keypoints를 받아서 Human3.6M format의 입력 및 출력 3D joint position들로써 사용합니다.

당신은 pre-trained model인 `PSTMOS_no_refine_48_5137_in_the_wild.pth`나 다음 명령어들을 사용해 직접 model을 학습시킴으로써 모델을 사용할 수 있습니다.

For the pre-training stage, please run:

```bash
python run_in_the_wild.py -k detectron_pt_coco -f 243 -b 160 --MAE --train 1 --layers 3 -tds 2 -tmr 0.8 -smn 2 --lr 0.0001 -lrd 0.97
```

For the fine-tuning stage, please run:

```bash
python run_in_the_wild.py -k detectron_pt_coco -f 243 -b 160 --train 1 --layers 3 -tds 2 --lr 0.0007 -lrd 0.97 --MAE_reload 1 --previous_dir your_best_model_in_stage_I.pth
```

이후에, 당신은 이 [레포지토리](https://github.com/zh-plus/video-to-pose3D) 를 사용해서 in-the-wild 비디오들로 우리 모델을 평가할 수 있습니다. 아래 명령어들을 따라와주세요

1. 코드를 설치하기 위해서 그들의 `README.md`를 따라가라
2. 그들 레포의 `checkpoint/` 폴더에 checkpoint를 위치시켜라.
3. 그들 레포의 root path에 `model/` 폴더와 `in_the_wild/videopose_PSTMO.py` 를 위치시켜라.
4. 그들 레포의 `common/` 폴더 안에 `in_the_wild/arguments.py`, `in_the_wild/generators.py`, 그리고 `in_the_wild/inference_3d.py` 을 위치시켜라
5. Run `videopose_PSTMO.py`!

Human3.6M 데이터셋의 프레임 속도는 50fps인 반면 대부분의 비디오는 25 또는 30fps입니다. 그래서 우리는 훈련시에는 tds=2로 테스팅 시에는 tds=1로 설정했습니다.

      
   - **2D Joint detectors**
     - Alphapose (Recommended)
       - Download **duc_se.pth** from ([Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)),
         place to `./joints_detectors/Alphapose/models/sppe`
       - Download **yolov3-spp.weights** from ([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)),
         place to `./joints_detectors/Alphapose/models/yolo`
     - HR-Net (Bad 3d joints performance in my testing environment)
       - Download **pose_hrnet*** from [Google Drive](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) | [Baidu pan](https://pan.baidu.com/s/1v6dov-TyPVOtejTNF1HXrA)), 
         place to `./joints_detectors/hrnet/models/pytorch/pose_coco/`
       - Download **yolov3.weights** from [here](https://pjreddie.com/media/files/yolov3.weights),
         place to `./joints_detectors/hrnet/lib/detector/yolo`
     - OpenPose (Not tested, PR to README.md is highly appreciated )
   - **3D Joint detectors**
      - Download **pretrained_h36m_detectron_coco.bin** from [here](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin),
        place it into `./checkpoint` folder
   - **2D Pose trackers (Optional)**
      - PoseFlow (Recommended)
        No extra dependences
      - LightTrack (Bad 2d tracking performance in my testing environment)
        - See [original README](https://github.com/Guanghan/lighttrack), and perform same *get started step* on `./pose_trackers/lighttrack`



## Usage

0. place your video into `./outputs` folder. (I've prepared a test video).

##### Single person video

1. change the `video_path` in the `./videopose.py`
2. Run it! You will find the rendered output video in the `./outputs` folder.

##### Multiple person video (Not implemented yet)

1. For developing, check `./videopose_multi_person`

   ```python
   video = 'kobe.mp4'
   
   handle_video(f'outputs/{video}') 
   # Run AlphaPose, save the result into ./outputs/alpha_pose_kobe
   
   track(video)					 
   # Taking the result from above as the input of PoseTrack, output poseflow-results.json # into the same directory of above. 
   # The visualization result is save in ./outputs/alpha_pose_kobe/poseflow-vis
   
   # TODO: Need more action:
   #  1. "Improve the accuracy of tracking algorithm" or "Doing specific post processing 
   #     after getting the track result".
   #  2. Choosing person(remove the other 2d points for each frame)
   ```




##### Tips
0. The [PyCharm](https://www.jetbrains.com/pycharm/) is recommended since it is the IDE I'm using during development.
1. If you're using PyCharm, mark `./joints_detectors/Alphapose`, `./joints_detectors/hrnet` and `./pose_trackers` as source root.
2. If your're trying to run in command line, add these paths mentioned above to the sys.path at the head of `./videopose.py`

## Advanced

As this script is based on the [VedioPose3D](https://github.com/facebookresearch/VideoPose3D) provided by Facebook, and automated in the following way:

```python
args = parse_args()

args.detector_2d = 'alpha_pose'
dir_name = os.path.dirname(video_path)
basename = os.path.basename(video_path)
video_name = basename[:basename.rfind('.')]
args.viz_video = video_path
args.viz_output = f'{dir_name}/{args.detector_2d}_{video_name}.gif'

args.evaluate = 'pretrained_h36m_detectron_coco.bin'

with Timer(video_path):
    main(args)
```

The meaning of arguments can be found [here](https://github.com/facebookresearch/VideoPose3D/blob/master/DOCUMENTATION.md), you can customize it conveniently by changing the `args` in `./videopose.py`.



## Acknowledgement

The 2D pose to 3D pose and visualization part is from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

Some of the "In the wild" script is adapted from the other [fork](https://github.com/tobiascz/VideoPose3D).

The project structure and `./videopose.py` running script is adapted from [this repo](https://github.com/lxy5513/videopose)



## Coming soon

The other feature will be added to improve accuracy in the future:

- [x] Human completeness check.
- [x] Object Tracking to the first complete human covering largest area.
- [x] Change 2D pose estimation method such as [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).
- [x] Test HR-Net as 2d joints detector.
- [x] Test LightTrack as pose tracker.
- [ ] Multi-person video(complex) support.
- [ ] Data augmentation to solve "high-speed with low-rate" problem: [SLOW-MO](https://github.com/avinashpaliwal/Super-SloMo).

