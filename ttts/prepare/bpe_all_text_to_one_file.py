import re
from pypinyin import pinyin, lazy_pinyin, Style

with open('data/bpe_train-set.txt', 'w') as out:
    #aishell3
    with open('data/label_train-set.txt', 'r') as f:
        for i,line in enumerate(f):
            if i<5:
                continue
            text = line.strip().split('|')[2].replace('% ','').replace('$','').replace('%','')
            pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
            out.write(pinyin+'\n')
    #data-baker 1w
    with open('data/000001-010000.txt', 'r') as f:
        for i,line in enumerate(f):
            if i%2==1:
                continue
            text = line.strip().split('\t')[1]
            text = re.sub(r'[#\d]', '', text)
            pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
            out.write(pinyin+'\n')

        