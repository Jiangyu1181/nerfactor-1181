## Preparation-1
0. Change dir and conda-env
   ```bash
   ssh jy@172.16.2.48
   cd /mnt/data2/jy/
   conda activate /mnt/data2/lcj/conda_envs/NeRFactor
    ```

1. Convert the dataset into our format:
    ```bash
    proj_root='/mnt/data2/jy/NeRFactor'
    repo_dir="$proj_root/nerfactor"
    indir="/mnt/data2/jy/datasets/nerfactor/brdf_merl"
    ims='512'
    outdir="/mnt/data2/jy/datasets/nerfactor/brdf_merl_npz/ims${ims}_envmaph16_spp1"
    REPO_DIR="$proj_root" "$proj_root"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"
    ```
   In this conversion process, the BRDFs are visualized to `$outdir/vis`, in the forms of characteristic clices and
   renders.

## Preparation-2

0. (Only once for all scenes) Learn data-driven BRDF priors (using a single GPU suffices):
    ```bash
    gpus='4'

    # I. Learning BRDF Priors (training and validation)
    gpus='6'
    ims='512'
    n_rays_per_step='1024'
    proj_root='/mnt/data2/jy/NeRFactor'
    repo_dir="$proj_root/nerfactor"
    data_root="/mnt/data2/jy/datasets/nerfactor/brdf_merl_npz/ims${ims}_envmaph16_spp1"
    outroot="$proj_root/output/train/merl${ims}rays${n_rays_per_step}"
    viewer_prefix=''
    REPO_DIR="$proj_root" "$proj_root/nerfactor/trainvali_run.sh" "$gpus" --config='brdf.ini' --config_override="n_rays_per_step=$n_rays_per_step,data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"

    # II. Exploring the Learned Space (validation and testing)
    ckpt="$outroot/lr1e-2/checkpoints/ckpt-50"
    REPO_DIR="$proj_root" "$proj_root/nerfactor/explore_brdf_space_run.sh" "$gpus" --ckpt="$ckpt"
    ```

1. Train a vanilla NeRF, optionally using multiple GPUs:
    ```bash
    scene='human'
    gpus='6'
    proj_root='/mnt/data2/jy/NeRFactor'
    repo_dir="$proj_root/nerfactor"
    viewer_prefix=''
    data_root="/mnt/data2/jy/datasets/nerfactor/rendered-images/$scene"
    near='1'
    far='10'
    lr='5e-4'
    imh='512'
    n_rays_per_step='1024'
    outroot="$proj_root/output/train/${scene}_nerf${imh}rays${n_rays_per_step}n${near}f${far}"
    REPO_DIR="$proj_root" "$proj_root/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="n_rays_per_step=$n_rays_per_step,data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"

    # Optionally, render the test trajectory with the trained NeRF, only can use 1 gpu
    gpus='3'
    scene='human'
    imh='512'
    near='1'
    far='10'
    proj_root='/mnt/data2/jy/NeRFactor'
    n_rays_per_step='1024'
    outroot="$proj_root/output/train/${scene}_nerf${imh}rays${n_rays_per_step}n${near}f${far}"
    lr='5e-4'
    ckpt="$outroot/lr$lr/checkpoints/ckpt-20"
    REPO_DIR="$proj_root" "$proj_root/nerfactor/nerf_test_run.sh" "$gpus" --ckpt="$ckpt"
    ```
   Check the quality of this NeRF geometry by inspecting the visualization HTML for the alpha and normal maps. You might
   need to re-run this with another learning rate if the estimated NeRF geometry is too off.

2. Compute geometry buffers for all views by querying the trained NeRF: (single GPU)
    ```bash
    scene='human'
    gpus='5'
    proj_root='/mnt/data2/jy/NeRFactor'
    repo_dir="$proj_root/nerfactor"
    viewer_prefix=''
    data_root="$proj_root/data/rendered-images/$scene"
    imh='512'
    lr='5e-4'
    near='1'
    far='10'
    n_rays_per_step='1024'
    trained_nerf="$proj_root/output/train/${scene}_nerf${imh}rays${n_rays_per_step}n${near}f${far}/lr${lr}"
    occu_thres='0.5'
    scene_bbox=''
    out_root="$proj_root/output/surf/$scene${imh}rays${n_rays_per_step}withoutmp4"
    mlp_chunk='375000' # bump this up until GPU gets OOM for faster computation
    REPO_DIR="$proj_root" "$proj_root/nerfactor/geometry_from_nerf_run.sh" "$gpus" --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk"
    ```
   For portability, this step runs sequentially, processing one view after another. If your infrastructure supports
   distributing jobs easily over multiple GPUs, you should consider having one GPU process one view to parallelize all
   views.

## Training, Validation, and Testing

Pre-train geometry MLPs (pre-training), jointly optimize shape, reflectance, and illumination (training and validation),
and finally perform simultaneous relighting and view synthesis (testing):

   # I. Shape Pre-Training， smooth normal and light visibility
   scene='human'
   gpus='4'
   model='nerfactor'
   overwrite='True'
   proj_root='/mnt/data2/jy/NeRFactor'
   repo_dir="$proj_root/nerfactor"
   viewer_prefix=''
   data_root="/mnt/data2/jy/datasets/nerfactor/rendered-images/$scene"
   near='1'
   far='10'
   imh='512'
   n_rays_per_step='1024'
   use_nerf_alpha='False'
   surf_root="$proj_root/output/surf/$scene${imh}rays${n_rays_per_step}"
   shape_outdir="$proj_root/output/train/${scene}_shape${imh}rays${n_rays_per_step}n${near}f${far}"
   imh='512'  # here must be 512, if not, will get error
   REPO_DIR="$proj_root" "$proj_root/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="n_rays_per_step=$n_rays_per_step,data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"
   
   # II. Joint Optimization (training and validation)  if get error, just use cpu
   scene='human'
   gpus='4'
   model='nerfactor'
   overwrite='True'
   proj_root='/mnt/data2/jy/NeRFactor'
   data_root="/mnt/data2/jy/datasets/nerfactor/rendered-images/$scene"
   repo_dir="$proj_root/nerfactor"
   viewer_prefix=''
   imh='512'
   n_rays_per_step='1024'
   brdf_ckpt="$proj_root/output/train/merl${imh}rays${n_rays_per_step}/lr1e-2/checkpoints/ckpt-50"
   near='1'
   far='10'
   use_nerf_alpha='True'
   xyz_jitter_std=0.01
   test_envmap_dir="/mnt/data2/jy/datasets/nerfactor/light-probes/test"
   shape_mode='finetune'
   shape_outdir="$proj_root/output/train/${scene}_shape${imh}rays${n_rays_per_step}n${near}f${far}"
   shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
   surf_root="$proj_root/output/surf/$scene${imh}rays${n_rays_per_step}"
   outroot="$proj_root/output/train/${scene}_$model${imh}rays${n_rays_per_step}n${near}f${far}"
   REPO_DIR="$proj_root" "$proj_root/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,xyz_jitter_std=$xyz_jitter_std,test_envmap_dir=$test_envmap_dir,shape_mode=$shape_mode,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"
   
   # III. Simultaneous Relighting and View Synthesis (testing)
   scene='human'
   gpus='6'
   model='nerfactor'
   imh='512'
   n_rays_per_step='1024'
   near='1'
   far='10'
   proj_root='/mnt/data2/jy/NeRFactor'
   outroot="$proj_root/output/train/${scene}_$model${imh}rays${n_rays_per_step}n${near}f${far}"
   ckpt="$outroot/lr5e-3/checkpoints/ckpt-10"
   color_correct_albedo='False'
   REPO_DIR="$proj_root" "$proj_root/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo="$color_correct_albedo" 

```

Training and validation (II) will produce an HTML of the factorization results:
normals, visibility, albedo, reflectance, and re-rendering. Testing (III) will produce a video visualization of the
scene as viewed from novel views and relit under novel lighting conditions.