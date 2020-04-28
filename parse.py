import r2pipe
import re
import glob
from tqdm import tqdm

regex = re.compile('[ ]{3,}[0-9A-Fa-f]{2,}')


def write(filename, label, data):
    with open(filename, 'w') as f:
        for file in data:
            for i in file:
                f.write('{} {}\n'.format(label, i))

def r2_setting(r2):
    r2.cmd('e asm.comments = 0')
    r2.cmd('e asm.xrefs = 0')
    r2.cmd('e asm.instr = 0')
    r2.cmd('e asm.offset = 0')
    r2.cmd('e asm.lines = 0')

def get_asm(r2, addr):
    r2.cmd("s {}".format(addr))
    ins = r2.cmd('pdr')
    asm = regex.findall(ins)
    # print(ins)
    # print(asm)
    asm = map(lambda x: x.lstrip(), asm)
    asm = ' '.join(asm)
    return asm



def get_data(filename):
    data = []
    r2 = r2pipe.open(filename)
    r2_setting(r2)
    r2.cmd('aa')
    afl = r2.cmd('aflq').strip().split('\n')
    afl = map(lambda x: x.strip(), afl)
    address = list(afl)
    for index, addr in enumerate(address):
        asm = get_asm(r2, addr)
        data.append(asm)

    r2.quit()
    return set(data)

def listfile(dir):
    return glob.glob(dir)

def main():
    archs = ['aarch64-rp3','alphaev56','alphaev67','armv8-rp3','avr','mips','mips64el','mipsel','nios2','powerpc','powerpc64','powerpc64le','riscv64','s390','s390x-64', 'sh','sparc','sparc64','x86_64-ubuntu18.04-linux-gnu','x86_64-ubuntu18.04-linux-gnu-static','xtensa']
    for arch in tqdm(archs):
        print("[{}] ....".format(arch))
        files = listfile(arch + '/*')
        data = []
        for file in tqdm(files):
            print("[{}] {}".format(arch, file))
            filedata = get_data(file)
            data.append(filedata)

        write('TRAIN_DATA/{}.train'.format(arch), arch, data)

if __name__ == "__main__":
    main()