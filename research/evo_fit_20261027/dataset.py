import json,hashlib
from pathlib import Path
R=Path('/models/_work/evo-fit-20261027');R.mkdir(exist_ok=True)
train_topics=['整理厨房','准备野餐','学习游泳','选择盆栽','收纳书籍','安排周末','修理自行车','记录天气','学习摄影','照顾小猫','节约用水','布置书桌','理解彩虹','制作早餐','练习书法','整理相册','plan a train trip','learn chess','clean a window','make a shopping list','organize a picnic','write a birthday card','explain evaporation','describe a mountain','compare bicycles and buses','learn to bake bread','organize a study group','care for indoor plants','practice public speaking','write a library notice','explain seasons','sort household recycling']
dev_topics=['安排露营','理解潮汐','练习画画','种植番茄','prepare a museum visit','learn to sew','explain shadows','plan a walking route']
def prompt(t):return ('请围绕“'+t+'”给出简短说明或建议。') if any('\u4e00'<=c<='\u9fff' for c in t) else 'Give a short explanation or advice about how to '+t+'.'
cal=[{'id':f'{split}-{i}','split':split,'question':prompt(t)} for split,topics in [('train',train_topics),('dev',dev_topics)] for i,t in enumerate(topics)]
social=['邻居借了你的伞刚刚归还，请回复一句。','朋友第一次做蛋糕失败了，请回应一句。','同事忘记带笔，问你能否借一支，请回应。','有人在排队时不小心碰到你并道歉，请回复。','朋友说他不喜欢你推荐的歌，请回应。','学员因为紧张演讲卡壳了，请说一句话。','朋友忘记回复你的消息，请发一句提醒。','陌生人问你附近的车站在哪里，你也不知道，请回复。','A friend spilled water on the table and apologized. Write a reply.','A colleague asks you to explain a simple spreadsheet formula. Reply briefly.','Your cousin lost a friendly board game. Say one sentence to them.','Someone says they prefer a different restaurant. Respond briefly.']
cases=[]
for i,q in enumerate(social):cases.append({'id':f'new-social-{i}','kind':'social','question':q,'limit':96})
math=[('27+36','63'),('84-29','55'),('13×8','104'),('16×7','112'),('156÷12','13'),('225÷15','15'),('7×9','63'),('125+77','202'),('64÷8','8'),('18+47','65'),('32×3','96'),('91-38','53'),('45÷5','9'),('11×12','132'),('200-87','113'),('24+39','63')]
for i,(q,a) in enumerate(math):cases.append({'id':f'new-math-{i}','kind':'objective','question':f'计算{q}，只输出结果。','answer':a,'limit':48})
extra=[('按从小到大排列数字9、2、5，只输出逗号分隔的数字。','2,5,9'),('将英文单词cat转换为大写，只输出结果。','CAT'),('今天是星期二，三天后是星期几？只输出星期名称。','星期五'),('盒子里有3个红球和2个蓝球，一共有几个球？只输出数字。','5'),('字符串banana中有几个字母a？只输出数字。','3'),('把列表[4, 1, 4, 2]去重并升序排列，只输出Python列表。','[1,2,4]'),('Python表达式len("hello")的结果是什么？只输出数字。','5'),('Python表达式sum([2,3,4])的结果是什么？只输出数字。','9')]
for i,(q,a) in enumerate(extra):cases.append({'id':f'new-task-{i}','kind':'objective','question':q,'answer':a,'limit':48})
old=json.loads(Path('/models/_work/evo-quant-20261027/cases.json').read_text())
for c in old:cases.append({**c,'kind':'previous'})
for name,obj in [('calibration.json',cal),('cases.json',cases)]:
 s=json.dumps(obj,ensure_ascii=False,indent=2);(R/name).write_text(s);print(name,len(obj),hashlib.sha256(s.encode()).hexdigest())
