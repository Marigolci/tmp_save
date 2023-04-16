
## 代码结构
运行代码的入口函数,由于大家的路径都不一样，可以每个人写自己的执行流程函数
* src/eval_pipeline.py 生成eval结果
* src/pred_pipeline.py 生成线上预估结果


## reuslt record
* 使用序列相邻结果做召回，使用lastitem作为trigger，加dhot召回补全
> 线上结果：0.2950302623415943，离线结果：0.24436948587630672

* 使用序列两两pair做统计，引入距离函数，使用lastitem作为trigger，加dhot召回补全
> 线上结果：0.30136，离线结果：0.252572890737683

* 使用序列两两pair做统计，引入距离函数，hot和i2i加权，使用lastitem作为trigger，加dhot召回补全  去除召回队列中在prev序列已出现的item
> 线上结果：0.34718，离线结果：0.29288041267089254

* 使用序列两两pair做统计，引入距离函数，hot和i2i和llr加权，使用lastitem作为trigger，加dhot召回补全  去除召回队列中在prev序列已出现的item
> 线上结果：-，离线结果：0.2941478507908295

* 使用序列两两pair做统计，引入距离函数，hot和i2i和llr加权，使用所有序列的item作为trigger，加dhot召回补全  去除召回队列中在prev序列已出现的item
> 线上结果：0.34991，离线结果：0.298279515796868

* 使用fasttext简单训练
> 线上结果：，离线结果：0.14357401999639197

* 使用fasttext简单训练, n_gram = 3
> 线上结果：，离线结果：0.13689082436746353

* fasttext调参，loss='hs', neg=100, minCountLabel=4, dim=300
> 线上效果：，离线结果：0.16769223914346093

* fasttext调参, loss='hs', neg=100, minCountLabel=4, dim=300, epoch=10
> 线上效果：，离线结果：0.17604385756418828
> 
* fasttext调参, loss='hs', neg=100, minCountLabel=4, dim=600, epoch=10
> 线上效果：，离线结果：0.18167009282328236

* 使用序列两两pair做统计，引入距离函数，hot和i2i和llr加权，使用所有序列的item作为trigger，加dhot召回补全, 加fasttext， 去除召回队列中在prev序列已出现的item
> 线上效果：0.346755，离线结果：0.2985722220175572

