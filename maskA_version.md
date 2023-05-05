# v1 版本
## 实现：
- 采样：每个timestep 的数据，以一定**概率** 对某一个agent进行mask处理
- contrastive loss 部分：
  - 实现三种representation 的计算方法 **待选**
  - loss计算使用curl 和 m-curl 中的cpc loss **待选**
### mask 之后得到表征的几种方法 : SG: stop gradient
MAT : 中是 obs -> CNNLayer_MAT -> Transformer_Layer_MAT -> representation 
#### method 1:  
    obs   -> CNNLayer_MAT           -> Transformer_Layer_MAT            -> rep
    label -> CNNLayer_MAT_target(SG)-> Transformer_Layer_MAT_target(SG) -> rep_target
#### method 2:  
    obs   -> CNNLayer_MAT           -> Transformer_Layer_MAT            -> Transformer_Layer_MASK -> rep
    label -> CNNLayer_MAT_target(SG)-> Transformer_Layer_MAT_target(SG)                           -> rep_target
#### method 3:
    obs   -> CNNLayer_MAT           -> Transformer_Layer_MASK           -> rep
    label -> CNNLayer_MAT_target(SG)->                                  -> rep_target

## TODO
- 确定representation的计算方法
- 确定表征学习的batch size
- 确定mask的概率
- 是否可以使用全局 observation 来恢复agent信息