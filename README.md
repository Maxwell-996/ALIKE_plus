本项目的主体框架参考自https://github.com/Shiaoming/ALIKE

总体说明

在项目中，本文整合了SuperPoint: Self-Supervised Interest Point Detection and Description中生成的几何图形数据集以及Homographic Adaptation方法，除此之外，本模型的网络结构参考自Breaking of brightness consistency in optical flow with a lightweight CNN network 旨在建立轻量级的特征点提取网络模型。



## 生成几何图形数据集

$\bullet$生成几何图形数据集参考自superpoint，对于生成图片的参数，可以在./magic_point_dataset/config.yaml中调整

$\bullet$由于同时运行具有torch和tensorflow的环境会导致gpu的显存资源占用异常的情况，建议使用离线生成数据集。 由于文件data_state.txt会一直维护最后生成的数据集，通过循环运行该脚本可以实现训练实时生成数据的效果。



```python
conda activate make_dataset
# 命令第一个参数是图片保存的路径 文件夹需预先创建 未创建的文件夹不会自动创建

python ./training/build_dataset.py /home/lizhonghao/ALIKE/magic_point_dataset/pic1

```
##  使用几何图形数据集对模型进行训练



```python
conda activate alike

python ./training/my_train.py 
```

### 实时查看模型效果
$\bullet$ 在训练时,除了观察loss之外，可以在"/home/lizhonghao/ALIKE/mgp_valres/"查看模型的效果，该路径可以在“training_wrapper.py”中修改，运行多个训练脚本时需要注意修改本路径，否则文件会覆盖存储。

文件夹下的内容组织形式如下：

epoch{epoch_id}_{pic_id}_ori.png表示原图

epoch{epoch_id}_{pic_id}_scoremap.png表示特征点检测的评分图

epoch{epoch_id}_{pic_id}_trans.png表示单应性变换后图片的特征点检测结果

epoch{epoch_id}_{pic_id}_.png无标注图片表示原图的特征点检测结果

### 项目结构

$\bullet$ xxNet.py :模型文件

$\bullet$ model_name.py: model_name对象继承自xxNet对象，交待了神经网络模型的解码过程。

$\bullet$ xx_wrapper.py :TrainWrapper对象继承自神经网络（model_name.py）对象，其交待了神经网络的训练细节，包括每个epoch中训练epoch和验证epoch的流程等，通过调用父类（model_name）中的extract()函数extract_dense_map()函数完成神经网络模型的forward过程。

$\bullet$ my_train.py/my_test.py :调用pytorch_lightning库提供模型生成的接口，pl.Trainer()函数进行训练时的信息记录和保存，模型的checkpoint保存在./training/log_my_train/ 中，文件的名字是训练的开始时间。神经网络的对象从xx_wrapper.py文件中生成，这里可以传入神经网络的规模参数（隐藏层维度等）,在该文件的第一行可以选取运行模型的gpu编号。

$\bullet$ homography_adaptation.py： 文件交待了对score_map的homography_adaptation过程。homography_adaptation过程首先对输入图片进行了单应性变换，通过多次调用神经网络的foward过程获得平均的score表示。

$\bullet$ hpatch.py/gen_mgp_data.py :模型的dataset类，分别对应hpatch数据集和几何图形数据集。

##  hpatch数据集生成伪标签

```python
conda activate alike

python ./training/my_test.py 
```
$\bullet$ checkpoint的选取在my_test.py的主函数第一行的参数“pretrained_model” ，伪标签的储存位置见test_warpper.py，在文件夹./hpatch_valres/中可以实时查看模型在验证集中的效果,包含特征点位置，scoremap等。


##  使用hpatch数据集对模型进行训练

$\bullet$ 使用hpatch数据集对模型进行训练（伪标签），需要修改./training/my_train.py最后面的train_loader参数，将train_loader设置为hpatch数据集的loader，然后执行如下所示的命令。

```python
conda activate alike

python ./training/my_train.py 
```
##  其他辅助脚本

$\bullet$ ./make_video.py :处理一系列图片生成视频，便于观察训练中提取的特征点的变化
