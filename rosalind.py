import re


path = 'D:/Download/SD模型/英文文章引言参考文献/scRNA_data/roseland_data'

with open(path + '/rosalind_gc.txt', 'r+') as f:
    context = f.readlines()
context = [i.strip() for i in context]


pattern = '^>.*'
def GC_content(seq):
    nametmp = ''
    tmp = []
    biggest = 0
    for i in seq:
        if re.findall(pattern, i): #第二次遇到title
            if tmp == []: #如果tmp为空
                tmp.append(i[1:]) #记录新序列
            else: #否则证明已有序列，计算前一个序列的GC含量
                pct = seq_calcu(''.join(tmp[1:]))
                if pct > biggest:
                    biggest = pct
                    nametmp = tmp[0]
                    tmp = [i[1:]]
                else:
                    tmp = [i[1:]]
        elif i == seq[-1]:
            tmp.append(i)
            pct = seq_calcu(''.join(tmp[1:]))
            if pct > biggest:
                biggest = pct
                nametmp = tmp[0]
        else:
            tmp.append(i)
    print(nametmp, biggest * 100)

def seq_calcu(fragment):
    count = 0
    content = ['G', 'C']
    for i in fragment:
        if i in content:
            count += 1
    return round(count / len(fragment), 8)
        
GC_content(context)