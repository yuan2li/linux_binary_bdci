import os
import time
import json
import argparse
from tqdm import tqdm
from LogRecorder import CLogRecoder

# import sys
# sys.setrecursionlimit(7000)

parser = argparse.ArgumentParser()
parser.add_argument('datapath', help='path to storage files')
parser.add_argument('funcfile', help='file to be processed')

ymd = time.strftime("%Y-%m-%d", time.localtime())
logger = CLogRecoder(logfile='%s.log' % (ymd))
logger.addStreamHandler()

# general-purpose registers
# https://cdrdv2.intel.com/v1/dl/getContent/671200
# 3.4.1 General-Purpose Registers
# without rex
x86_byte_reg = ['al', 'bl', 'cl', 'dl', 'ah', 'bh', 'ch', 'dh']
x86_word_reg = ['ax', 'bx', 'cx', 'dx', 'di', 'si', 'bp', 'sp']
x86_dword_reg = ['e' + reg for reg in x86_word_reg]

# with rex
x86_byte_high = ['dil', 'sil', 'bpl', 'spl']
x86_r_reg = ['r' + str(i) for i in range(8, 16)]
x86_qword_reg = ['r' + reg for reg in x86_word_reg] + x86_r_reg

x86_byte_reg += x86_byte_high + [reg + 'b' for reg in x86_r_reg]
x86_word_reg += [reg + 'w' for reg in x86_r_reg]
x86_dword_reg += [reg + 'd' for reg in x86_r_reg]

# https://developer.arm.com/documentation/dui0801/a/
# General-purpose registers in AArch32 state
arm32_reg = ['r' + str(i) for i in range(15)]
# Registers in AArch64 state
arm64_reg = [j + str(i) for j in ('x', 'w') for i in range(31)]
# https://azeria-labs.com/arm-data-types-and-registers-part-2/
arm_alias_reg = ['fp', 'sp', 'lr', 'pc']

registers = {
    'x86': x86_byte_reg + x86_word_reg + x86_dword_reg + x86_qword_reg,
    'arm': arm32_reg + arm64_reg + arm_alias_reg,
    'mipsel': []
}

# https://www.ic.unicamp.br/~ranido/mc404/docs/ARMv7-cheat-sheet.pdf
# https://uweb.engr.arizona.edu/~ece369/Resources/spim/MIPSReference.pdf
# https://www.cs.uaf.edu/2011/spring/cs641/lecture/02_10_assembly.html

tran_ins = {
    'x86': ['mov', 'push', 'pop', 'xchg', 'in', 'out', 'xlat', 'lea', 'lds', 'les', 'lahf', 'sahf', 'pushf', 'popf'],
    'arm': ['b', 'bal', 'bne', 'beq', 'bpl', 'bmi', 'bcc', 'blo', 'bcs', 'bhs', 'bvc', 'bvs', 'bgt', 'bge', 'blt', 'ble', 'bhi', 'bls'],
    'mipsel': ['beqz', 'beq', 'bne', 'bgez', 'b', 'bnez', 'bgtz', 'bltz', 'blez', 'bgt', 'bge', 'blt', 'ble', 'bgtu', 'bgeu', 'bltu', 'bleu']
}

call_ins = {
    'x86': ['call'],
    'arm': ['bl'],
    'mipsel': ['jalr']
}

arith_ins = {
    'x86': ['add', 'adc', 'adcx', 'adox', 'sbb', 'sub', 'mul', 'div', 'inc', 'dec', 'imul', 'idiv', 'cmp', 'neg', 'daa', 'das', 'aaa', 'aas', 'aam', 'aad'],
    'arm': ['add', 'adc', 'qadd', 'sub', 'sbc', 'rsb', 'qsub', 'mul', 'mla', 'mls', 'umull', 'umlal', 'smull', 'smlal', 'udiv', 'sdiv', 'cmp', 'cmn', 'tst'],
    'mipsel': ['add', 'addu', 'addi', 'addiu', 'and', 'andi', 'div', 'divu', 'mult', 'multu', 'slt', 'sltu', 'slti', 'sltiu']
}


class AttributesBlockLevel(object):
    def __init__(self, func):
        self._code = func['code']
        self._arch = func['arch'].split('_')[0]
        self._blocks = [str(block[0]) for block in func['block']]

        self._CFG = {}
        for blkID in self._blocks:
            self._CFG[blkID] = []
        for cfg in func['cfg']:
            self._CFG[str(cfg[0])].append(cfg[1])

        # logger.INFO('computing offspring...')
        # self._offspring = {}
        # self.visit = set()
        # for node in self._blocks:
        #     self.visit = set()
        #     self._offspring[node] = self.dfs(node)

    # def dfs(self, node):
    #     if node in self.visit:
    #         return 0
    #     self.visit.add(node)
    #     offspring = 0
    #     for succ_node in self._CFG[node]:
    #         if succ_node not in self.visit:
    #             offspring += self.dfs(succ_node) + 1
    #     # logger.INFO('node %s returns offspring %d' % (node, offspring))
    #     return offspring

    def is_string(self, opr):
        if self._arch == 'mipsel' and opr[0] == '$':
            return False
        else:
            if opr in registers[self._arch]:
                return False
            else:
                opr = opr.replace('_', '')
                if opr.isalpha():
                    return True
                return False

    def is_number(self, opr):
        if self._arch == 'arm':
            if opr[0] == '#':
                return True
            return False
        try:
            if any([opr.startswith(p) for p in ('0x', '-0x')]):
                int(opr, 16)
            else:
                int(opr)
            return True
        except:
            return False
    
    def get_cfg(self):
        cfg_pairs = sorted(self._CFG.items(), key=lambda x: x[0])
        cfg_vals = [cp[1] for cp in cfg_pairs]
        return cfg_vals

    def get_att_block(self, block):
        blkID, startOA, endOA = block[0], block[1], block[2]
        dic = {}
        block_codes = [code[-1] for code in self._code if startOA <= code[0] < endOA]
        opcodes = [code.split('\t')[0] for code in block_codes]
        operands = [code.split('\t')[1] for code in block_codes if '\t' in code]
        operands = sum([opr.split(',') for opr in operands], [])
        operands = [opr.strip() for opr in operands]

        dic['string_constants'] = len([opr for opr in operands if self.is_string(opr)])
        dic['numeric_constants'] = len([opr for opr in operands if self.is_number(opr)])
        dic['no_trans'] = len([opc for opc in opcodes if opc in tran_ins[self._arch]])
        dic['no_calls'] = len([opc for opc in opcodes if opc in call_ins[self._arch]])
        dic['no_ins'] = len(opcodes)
        dic['no_ariths'] = len([opc for opc in opcodes if opc in arith_ins[self._arch]])
        # dic['no_offsprings'] = self._offspring[str(blkID)]
        dic['no_offsprings'] = len(self._CFG[str(blkID)])

        # 'no_ariths', 'no_calls', 'no_ins', 'no_offsprings', 'no_trans', 'numeric_constants', 'string_constants'
        att_pairs = sorted(dic.items(), key=lambda x: x[0])
        att_vals = [ap[1] for ap in att_pairs]
        return att_vals


class ExtractFeature(object):
    def __init__(self, datapath, funcfile):
        self._funcfile = datapath + os.sep + funcfile
        self._outfile = datapath + os.sep + funcfile.split('.')[0] + '.feature.json'

    def get_func_file(self):
        return self._funcfile

    def get_out_file(self):
        return self._outfile

    # read json file to get features
    def _readfiles(self):
        with open(self._funcfile) as f:
            for line in f:
                # print line
                x = json.loads(line.strip())
                yield x

    def get_function_feature(self, size):
        functions = self._readfiles()
        with tqdm(total=size, ncols=100, desc="feature extracting") as pbar:
            for func in functions:
                # logger.INFO('fid: ' + str(func['fid']))
                dic = {}
                dic['src'] = func['arch'] + '_' + func['compiler'] + '_' + func['opti']
                blocks = func['block']
                dic['n_num'] = len(blocks)

                AB = AttributesBlockLevel(func)
                dic['succs'] = AB.get_cfg()
                dic['features'] = [AB.get_att_block(blk) for blk in blocks]

                dic['fname'] = func['fid']
                with open(self._outfile, 'a') as f:
                    json.dump(dic, f, ensure_ascii=False)
                    f.write('\n')
                pbar.update(1)


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.datapath # E:/Dataset/linux_binary
    func_file = args.funcfile # train.func.json test.func.update.json
    data_type = func_file.split('.')[0]
    data_size = {
        'train': 2270931,
        'test': 627566,
        'demo': 2
    }

    gf = ExtractFeature(data_path, func_file)

    logger.INFO(gf.get_func_file())

    feature = gf.get_function_feature(data_size[data_type])

    logger.INFO(gf.get_out_file())
    logger.INFO("\n---------------------\n")
