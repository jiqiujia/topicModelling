# encoding=utf-8
from __future__ import unicode_literals, print_function, division


import re
import jpype
from term import Term

pattern_4byte_char = re.compile(u'[^\u0000-\uD7FF\uE000-\uFFFF]', re.UNICODE)


class HanlpStandardTokenizer:
    """
    Hanlp分词的封装类
    TODO: 自定义启用hanlp分词器功能如人名地名识别等
    """

    def __init__(self, path_class, data_dir='', remove_4byte_char=True):
        """
        初始化
        :param path_class: jar包、资源文件路径，示例如下
            path_class_win = "-Djava.class.path=C:\hanlp\hanlp-1.2.8.jar;C:\hanlp"
            path_class_nix = "-Djava.class.path=/Users/nathanlvzs/data/hanlp/hanlp-1.3.4.jar:/Users/nathanlvzs/data/hanlp"
        :param data_dir: 数据文件夹的路径，即包含data子文件夹的文件夹路径，
            仅在使用dphanlp时指定该参数，此时path_class中指定jar即可
        """
        if not jpype.isJVMStarted():
            print('JVM started! Closing...')
            #self.dispose()
            jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', path_class, "-Xms1g", "-Xmx2g")
        print(jpype.getClassPath())
        print(jpype.isJVMStarted())
        self.tokenizer = jpype.JClass('com.hankcs.hanlp.HanLP')
        #self.CRFSegment = jpype.JClass('com.hankcs.hanlp.seg.CRF.CRFSegment')()
        self.stop_word_dictionary = jpype.JClass('com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary')
        self.remove_4byte_char = remove_4byte_char
        # if data_dir:
        #     hanlp_config = jpype.JClass('com.hankcs.hanlp.HanLP$Config')
        #     hanlp_config.LoadResouce(data_dir)
        pass

    def _clean(self, line):
        # reference:
        # https://stackoverflow.com/questions/3220031/how-to-filter-or-replace-unicode-characters-that-would-take-more-than-3-bytes
        if self.remove_4byte_char:
            return pattern_4byte_char.sub('', line)
        else:
            return line

    def cut_with_nature(self, line, rm_stop_word=False):
        """
        分词
        :param line: 输入文字
        :return: Term列表，Term实例包含word和nature两个成员
        :param rm_stop_word: 是否去除停留词
        """
        line = self._clean(line)
        result = []
        raw_result = self.tokenizer.segment(jpype.JString(line))
        if rm_stop_word:
            self.stop_word_dictionary.apply(raw_result)
        # for token in self.tokenizer.segment(JString(line)):
        # <class 'jpype._jclass.java.util.ArrayList'>
        # result.append(Term(token.word, token.nature.toString()))
        for token in raw_result:
            # <class 'jpype._jclass.java.util.ArrayList'>
            result.append(Term(token.word, token.nature.toString()))
        return result

    # def cut_with_crf(self, line, rm_stop_word=False):
    #     line = self._clean(line)
    #     result = []
    #     raw_result = self.CRFSegment.seg(jpype.JString(line))
    #     if rm_stop_word:
    #         self.stop_word_dictionary.apply(raw_result)
    #     for token in raw_result:
    #         # <class 'jpype._jclass.java.util.ArrayList'>
    #         result.append(Term(token.word, token.nature.toString()))
    #     return result
    
    def cut(self, line, rm_stop_word=False):
        """
        分词
        :param line: 输入文字
        :return: 分词结果字符串列表
        :param rm_stop_word: 是否去除停留词
        """
        # 对于“Let's Go!!!🍙🍰🍜🍟🍢🍣🍕🍤......”切词会出HeapCorruption的异常，程序直接挂
        # Python3 jpype exit code -1073740940
        # emoji unicode字符集：http://unicode.org/Public/emoji/11.0/
        # 单独切，一两次没问题，重复几次就有问题
        # 在java端测试并没有出错
        # 4byte字符没有被准确转换？暂且通过_clean方法去除4byte字符
        # multi-byte characters between Java and Python
        # line = self._clean(line)
        # result = []
        # for token in self.tokenizer.segment(JString(line)):
        #     result.append(token.word)
        result = []
        for term in self.cut_with_nature(line, rm_stop_word):
            result.append(term.word)
        return result

    def add_custom_words(self, words):
        custom_dictionary = jpype.JClass('com.hankcs.hanlp.dictionary.CustomDictionary')
        for word in words:
            word = self._clean(word)
            custom_dictionary.add(jpype.JString(word))

    def add_stop_words(self, words):
        for word in words:
            word = self._clean(word)
            self.stop_word_dictionary.add(word)
        pass

    def dispose(self):
        """
        销毁JVM资源，调用后，就无法再使用segment方法了
        :return:
        """
        jpype.shutdownJVM()


if __name__ == '__main__':
    hanlp_path = "-Djava.class.path=hanlp-1.6.8.jar;."
    segmenter = HanlpStandardTokenizer(hanlp_path, data_dir="E:\\libs\\dphanlp\\data")
    segmenter.add_stop_words(['亚洲'])
    segmenter.add_custom_words(['必去的'])
    text = "【吃货攻略】必去的亚洲美食集市觅食之旅🍙🍰🍜🍟🍢🍣🍕🍤......"
    for token in segmenter.cut(text, True):
        print(token)
    pass
