# NER
一个中文的实体命名识别系统

当前版本基于双向循环神经网络（BiRNN） + 条件随机场（CRF）来完成实体的标注。 基本思路是利用深度神经网络提取特征，从而避免了手动提取特征的麻烦。
第二部和传统的方式一样，使用CRF在做最后的标注。

该程序使用Tensorflow完成，使用了当前较新的DataSet API，使数据预处理和feed更优雅。

由于使用了新的API， Tensorflow版本必须大于1.2.0，代码升级后使用的是TensorFlow1.4，不确定原来Tensorflow1.2是否还兼容。

如何使用？

    1. 建议安装tensorflow >= 1.4.1
    
    2. 提供训练所需的数据，具体格式在resource文件夹里有展示。但是需要自行分词。只需提供3个文件：
        source.txt target.txt 和 预训练的词向量。
        
    3. 训练词向量，训练工具可以是gensim的word2vector或者glove等等，然后将词和对应的词向量以以下格式保存。
        具体格式是： 
        单词A:0.001,0.001,0.001,....
        单词B:0.001,0.001,0.001,....
        单词C:0.001,0.001,0.001,....
        .
        .
        .
        有些训练工具得出的文件结果就是以上格式不需要修改．　程序默认embedding size是300, 可以按需求做更改
        （注意：训练词向量的数据量越大越好，不只限于当前语聊，如果需要训练好的词向量可以联系我。）
        
    4. 修改config.py里的文件存路径，所有的配置都在这个文件里。
    
    5. 训练：修改config.py
        tf.app.flags.DEFINE_string("action", 'train', "train | predict")
        $ python rnn.py
        
    6. 预测：修改config.py
        tf.app.flags.DEFINE_string("action", 'predict', "train | predict")
        $ python rnn.py
       

注意：
    原本resource文件中只包含predict.txt, source.txt, target.txt, 如果更换自己的词向量文件记得删除其他自动生成的文件。

更多文件、问题见issues. 

