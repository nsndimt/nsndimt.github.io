## Indri踩坑

1. 在[Field](https://sourceforge.net/p/lemur/wiki/Fields/)中dog.title,header被描述为出现在title或header中的dog，但实际上根据[reference](https://sourceforge.net/p/lemur/wiki/Indri%20Query%20Language%20Reference/)中的描述title和dog并非或的关系而是与关系
2. trectext不一定要按照doc docno的格式可以自定义格式 [参考](https://sourceforge.net/p/lemur/discussion/546029/thread/cea06f6e/) [好像不会自动添加](https://sourceforge.net/p/lemur/discussion/546029/thread/aa0b0575/?limit=25#0ca5)
3. 在indri5.10时曾经遇到baseline选项不能为okapi的情况，5.11好像修复了 P.S indri介绍baseline选项的文件不在wiki里在安装文件夹的doc文件中的index里
4. [beta与dirichlet](http://www.52nlp.cn/lda-math-%E8%AE%A4%E8%AF%86betadirichlet%E5%88%86%E5%B8%831)[beta与dirichlet2](http://www.52nlp.cn/lda-math-%E8%AE%A4%E8%AF%86betadirichlet%E5%88%86%E5%B8%832)[beta与dirichlet3](http://www.52nlp.cn/lda-math-%E8%AE%A4%E8%AF%86betadirichlet%E5%88%86%E5%B8%833)后验=先验\*似然 若先验后验形式一致则称共轭，但是似然不是教课书上的分布，因为其数据已知而参数未知，与教科书相反。
5. [beta与dirichlet期望计算](http://xinsong.github.io/2014/04/29/beta/)

## Mose踩坑
boost在centos6上编译有坑，原因是boost在gcc4.4 .7下编译需要打个[patch](https://svn.boost.org/trac/boost/ticket/11856) P.S red hat centos可以认为是同一个系统。谷歌编译错误要记住两点一在出错时退出二记下错误代号错误发生地点;谷歌时加上gcc版本、系统版本有时有奇效。

编译boost命令
```       
/root/boost_1_60_0/bootstrap.sh 
/root/boost_1_60_0/b2 -q install
#b2-q选项代表遇到错误退出
```

[irstlm的compile-lm的选项语法](https://github.com/irstlm-team/irstlm/issues/2)

[irstlm已经转到github了，虽然谷歌第一位是sf](https://github.com/irstlm-team/irstlm)

[非常好的摩西教程](http://blog.csdn.net/han_xiaoyang/article/details/10101701)  

完整运行命令  
```
/root/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 9 align s t  align.clean 1 80
/root/irstlm/bin/add-start-end.sh <align.t >align.sb.t
export IRSTLM=/root/irstlm/
/root/irstlm/bin/build-lm.sh -i align.sb.t -p -s improved-kneser-ney -o align.lm.t --debug
#脚本只有在DEBUG模式下才能正常运行
/root/irstlm/bin/compile-lm -text=yes align.lm.t.gz align.arpa.t
/root/mosesdecoder/bin/build_binary -i align.arpa.t align.blm.t
/root/mosesdecoder/scripts/training/train-model.perl -cores 24 -root-dir /root/mosesdecoder/ -corpus /root/mosesdecoder/align.clean -f s -e t -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:/root/mosesdecoder/align.blm.t -external-bin-dir /root/giza-pp/
/root/mosesdecoder/scripts/training/mert-moses.pl /root/mosesdecoder/align.clean.s /root/mosesdecoder/align.clean.t /root/mosesdecoder/bin/moses /root/mosesdecoder/model/moses.ini  --mertdir /root/mosesdecoder/bin/ --decoder-flags="-threads 24"
/root/mosesdecoder/bin/moses -f /root/mosesdecoder/model/moses.ini
#多用绝对地址但有时如果示例里用相对地址，那可能作者并没有考虑绝对地址的情况
#linux中 &> 与 >& 等价，但是后一种真的很少见
```  

每次实验都要删除所有中间输出否则可能出现如下错误  
```
ERROR: target word 118049 is not in the vocabulary list
WARNING: The following sentence pair has source/target sentence length ration more than the maximum allowed limit for a source word fertility source length = 1 target length = 11 ratio 11 ferility limit : 9
WARNING: sentence 2176 has alignment point (3, 3) out of bounds (6, 3)
<<<<<<< HEAD
```  

如果mert-moses.pl脚本中出错显示
```
exec: /root/mosesdecoder/scripts/training/filter-model-given-input.pl ./filtered /root/mosesdecoder/train1/model/moses.ini /root/mosesdecoder/align.clean.s1v
Executing: /root/mosesdecoder/scripts/training/filter-model-given-input.pl ./filtered /root/mosesdecoder/train1/model/moses.ini /root/mosesdecoder/align.clean.s1v > filterphrases.out 2> filterphrases.err
Exit code: 1
ERROR: Failed to run '/root/mosesdecoder/scripts/training/filter-model-given-input.pl ./filtered /root/mosesdecoder/train1/model/moses.ini /root/mosesdecoder/align.clean.s1v'. at /root/mosesdecoder/scripts/training/mert-moses.pl line 1748.
```  

很可能是缺少工作路径需要加上--working-dir /root/mosesdecoder/train1[出处](http://blog.sciencenet.cn/blog-200204-205469.html)

[加入cmph支持](http://www.statmt.org/moses/?n=Advanced.RuleTables)[cmph安装](https://github.com/zvelo/cmph/blob/master/INSTALL)
