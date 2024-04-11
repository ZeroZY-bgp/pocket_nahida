from utils import save_json, load_json

kb_n_lst = load_json("kb/keyword.json")

print(f"KB keywords len: {len(kb_n_lst)}")

n_lst = [
    '旅行者',
    '世界树',
    '禁忌知识',
    '蒙德',
    '须弥',
    '璃月',
    '稻妻',
    '枫丹',
    '天空岛',
    '纳塔',
    '布耶尔',
    '纳西妲',
    '巴巴托斯',
    '温迪',
    '摩拉克斯',
    '钟离',
    '巴尔泽布',
    '巴尔',
    '雷电真',
    '雷电影',
    '芙宁娜',
    '芙卡洛斯',
    '厄歌莉娅',
    '散兵',
    '流浪者',
    '教令院',
    '明论派',
    '素论派',
    '知论派',
    '因论派',
    '妙论派',
    '生论派',
    '甘露花海',
    '须弥城'
]
print(f"Manual keywords len: {len(n_lst)}")

n_lst.extend(kb_n_lst)

n_lst = list(set(n_lst))  # 去重

print(f"Final keywords len: {len(n_lst)}")

save_json('kb/word.json', n_lst)

for n in n_lst:
    print(n)
