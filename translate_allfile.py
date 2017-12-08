'''
Translates a source file using a translation model.
'''
import argparse

import os
import datetime
import numpy
import sys
import subprocess


class CmdParser:
    def __init__(self, args):
        self.args = args

    def to_bleu_cmd(self, src, saveto, ref):
        return 'perl %s %s < %s' % (self.args['multi_bleu_perl'], ref, saveto+'.lctok')
        #return 'python %s -c /home/zhaocq/mt/abstractNMT/chunkbased/mteval/mteval-v13a.pl -s %s -r %s -t %s' \
        #   % (self.args['multi_bleu_perl'],
        #       src, 
        #       ref,
        #       saveto)
        

    def to_translate_cmd(self, source, saveto, use_gpu):
        '''

        :param model_file:
        :return: cmd
        '''

        # if not os.path.exists(source):
        #     return (source, ''), "echo"
        # if 'iter' in self.args['model']:
        #     iter_index = self.args['model'].index('iter')
        #     iter = int(self.args['model'][iter_index+4: self.args['model'].index('.', iter_index)])
        #     saveto = saveto + '.iter' + str(iter)
        # else:
        #     saveto = saveto + '.final'

        # if self.args['n']:
        #     return (source, saveto), "THEANO_FLAGS='device=%s,floatX=%s' python %s -n -m %s %s %s %s %s %s " \
        #         % (use_gpu,
        #            self.args['floatX'],
        #            self.args['script'],
        #            self.args['pkl_model'],
        #            self.args['model'],
        #            self.args['dictionary'],
        #            self.args['dictionary_target'],
        #            source,
        #            saveto)
        # else:
        return (source, saveto), "THEANO_FLAGS='device=%s,floatX=%s' python %s %s %s %s %s %s " \
            % (use_gpu,
               self.args['floatX'],
               self.args['script'],
               self.args['model'],
               self.args['dictionary'],
               self.args['dictionary_target'],
               source,
               saveto)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # k: beam size
    # parser.add_argument('-k', '--beam-size', type=int, default=5)
    # gpu
    parser.add_argument('-g', '--gpu', type=str, default='gpu0')
    # floatX
    parser.add_argument('-f', '--floatX', type=str, default='float32')
    # n: if normalize
    # parser.add_argument('-n', action="store_true", default=False)
    # max processes
    parser.add_argument('-l', '--process-limit', type=str, default='5')
    # multi-bleu.per script
    parser.add_argument('-b', '--multi-bleu-perl', type=str)
    # script
    parser.add_argument('-s', '--script', type=str)
    # model pkl
    # parser.add_argument('-p', '--pkl-model', type=str)
    # model file
    parser.add_argument('-m', '--model', type=str)
    # source side dictionary
    parser.add_argument('dictionary', type=str)
    # target side dictionary
    parser.add_argument('dictionary_target', type=str)
    # source file
    parser.add_argument('sources', type=str)
    # reference
    parser.add_argument('refs', type=str)
    # translation file
    parser.add_argument('savetos', type=str)

    parser.add_argument('save_bleu', type=str)

    args = parser.parse_args()

    cmd_parser = CmdParser(dict(vars(args).items()))

    start_time = datetime.datetime.now()

    sources = args.sources.split(',')
    refs = args.refs.split(',')
    savetos = args.savetos.split(',')

    assert len(sources) == len(refs)
    assert len(sources) == len(savetos)

    limit = 0
    popens = []
    use_gpu_list = []
    for gpu, gpu_limit in zip(args.gpu.split(','), args.process_limit.split(',')):
        gpu_limit = int(gpu_limit)
        for _ in xrange(gpu_limit):
            use_gpu_list.append(gpu)
        limit += gpu_limit

    times = len(sources) / limit
    use_gpu_list *= (times + 1)

    bleu_source_list = []
    bleu_list = []
    source_list = []
    while True:
        if len(popens) < limit and len(sources) > 0:
            source = sources.pop(0)
            saveto = savetos.pop(0)
            use_gpu = use_gpu_list.pop(0)
            source_saveto, cmd = cmd_parser.to_translate_cmd(source, saveto, use_gpu)
            source_list.append(source_saveto)
            popens.append(subprocess.Popen(cmd,
                stdout=subprocess.PIPE, shell=True))
        elif len(popens) == 0:
            break
        else:
            popen = popens.pop(0)
            source_saveto = source_list.pop(0)
            ref = refs.pop(0)
            popen.wait()
            if source_saveto[1] == '': continue;
            print popen.stdout.readlines()[-1].strip()
            # run multi-bleu.perl and get the BLEU result
            popen = subprocess.Popen(cmd_parser.to_bleu_cmd(source_saveto[0], source_saveto[1], ref),
                stdout=subprocess.PIPE, shell=True)
            bleu_source_list.append(source_saveto[0])
            popen.wait()
            ret = popen.stdout.readline().strip()
            try:
                # bleu = float(ret)
                bleu = float(ret[7:ret.index(',')])
            except ValueError:
                bleu = -1.
            bleu_list.append(bleu)

    with open(args.save_bleu, 'w') as fp:
        fp.write('from %s ......\n' % args.model)
        for source in bleu_source_list:
            fp.write("%s\n" % (source))
        for bleu in bleu_list:
            fp.write("%f  " % (bleu))
        fp.write("\n")

    print 'Total Elapsed Time: %s' % str(datetime.datetime.now() - start_time)
