# 一种基于双重语义协作网络的图像描述方法(An image captioning method based on DSC-Net)

这个库包括论文“[一种基于双重语义协作网络的图像描述方法](https://kns.cnki.net/kcms2/article/abstract?v=HlDkjiDVjGtKgx1uqp8L8T5OGlr0Hnm2050at2-Wvo7hslC2y3VAznVP-j-jDFtn0kzzx4Rv5LF8lqHkxbcMMKAOpQ43jNM8vAx8olEUszuzDd2JveOmhYQScdpY2wSG3PwWRtn46wH-M_aTisB5wy8t4TTXDNktY916SXm78XwfkJmM90o-HA==&uniplatform=NZKPT&language=CHS)”的参考代码。



## 实验环境搭建

Most of the previous works follow [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer), but they utilized some lower-version packages. Therefore, we recommend  referring to [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx). 其中下载en_core_web_sm时，如果遇到无法解决的网络问题，可以采用[这里](https://github.com/luo3300612/Transformer-Captioning)对应文件中的代码以跳过调用spacy包的步骤

## 数据准备

* donannotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing). Extarct and put it in the project root directory. 
* **Feature**. We extract feature with the code in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). You can download the features we used [here](https://github.com/luo3300612/image-captioning-DLCT).
* **evaluation**. We use standard evaluation tools to measure the performance of the model, and you can also obtain it [here](https://github.com/luo3300612/image-captioning-DLCT). Extarct and put it in the project root directory.

## Training

```python
python train.py --exp_name dlct --batch_size 50 --head 8 --features_path coco_all_align.hdf5 --annotation m2_annotations --workers 5 --rl_batch_size 100 --image_field ImageAllFieldWithMask --model DLCT --rl_at 17 --seed 118
```

## Evaluation

```python
python eval.py --annotation annotation --workers 5 --features_path coco_all_align.hdf5 --model_path saved_models/pretrained_model.pth --model DLCT --image_field ImageAllFieldWithMask --grid_embed --box_embed --dump_json gen_res.json --beam_size 5
```

## References

[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

[3] [DLCT](https://github.com/luo3300612/image-captioning-DLCT)

## Acknowledgements

感谢 [M2](https://github.com/aimagelab/meshed-memory-transformer) 的开创性贡献与 [dlct](https://github.com/luo3300612/image-captioning-DLCT) 提供的代码框架，这些内容启发了我们的工作。
