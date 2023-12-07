# Translator-Transformer

## 简介
一个使用`Pytorch` 构建的 `Transformer 架构` 英译中翻译程序 <br>

## 项目结构 
`setting.py`:模型相关参数，文件目录的配置文件。  
`utils.py`:一些工具函数。  
`data_pre.py`:数据的预处理，得到输出模型的batch数据和相关的mask矩阵  
`model.py`:模型文件。  
`train.py`:进行模型的训练。和最好模型的保存。  
`test.py`:对测试集句子的测试输出。  
`bleu_score.py`:对机器翻译评分。  
`infer.py`:实现单个句子进行翻译。  
`app.py`:通过使用infer.py封装的单个句子翻译的方法，实现flask api  


## 如何使用
 
1. **下载nltk依赖**

如果之前已经下载过，这一步可以跳过
```python
import nltk
nltk.download('punkt')
```
2. **数据处理/模型训练**
```
python train.py
```
 训练好的模型会存在 ./save 文件夹下
 
3. **模型交互**

```
python infer.py --sentence="I love you" 
```
4. **Flask API**

启动flask服务
```
python app.py
```
flask 接口  `/translation` post 方法
```json
{
  "sentence": "英文句子"
}
```
返回
```json
{
  "result": "翻译结果",
  "msg": "success",
  "code": 200
}
```

## 模型训练数据
使用**14533**条翻译数据进行训练。  
数据文件格式：en`\t`cn

    Anyone can do that.	任何人都可以做到。
    How about another piece of cake?	要不要再來一塊蛋糕？
    She married him.	她嫁给了他。
    I don't like learning irregular verbs.	我不喜欢学习不规则动词。
  

## 结果评估
使用BLEU算法进行翻译效果评估
BLEU算法评价结果：  
    
    对399条翻译句子效果进行评估
    验证集:0.1075088492716548，n-gram权重：(1,0,0,0)
          0.03417978514554449,n-gram权重：(1,0.2,0,0)

## 参考文献
1. [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
2. [HarvardNLP "The Annotated Transformer"](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [Transformer 代码完全解读](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/120540057)
4. [Attention专场](https://blog.csdn.net/u012759262/article/details/103999959)
5. [taoztw/Transformer](https://github.com/taoztw/Transformer)
