import jieba
from cutword import Cutter
import time

text = '''
密云区国资委坚持把开展主题教育同推动区属国有企业中心工作紧密结合起来，指导区属国有企业深入开展理论学习、调查研究、深化重点领域改革、解决群众急难愁盼问题，切实为奋进新征程、建功新时代凝心聚力。

聚焦学懂弄通，抓好理论学习。第一时间为全系统党员干部配发学习资料，各基层党组织利用“三会一课”、学习强国、主题党日等多种形式开展理论学习。

聚焦重点难点，深入开展调研。紧扣国资国企改革、重大项目建设、本部门年度目标任务，全系统处级领导干部领题选题三十余个，并围绕调研选题深入开展调研。

聚焦推动高质量发展，深化重点领域改革。利用信息化手段提高监管能力，不断完善“智慧国资”动态监管平台。研究拟定《密云区盘活闲置资产三年行动计划》，努力提高国有资本运营效率和质量。制定《区属国有企业组建混合所有制企业的实施意见》，稳步推进混改工作。

聚焦暖民心、解民忧，积极解决群众急难愁盼问题。为民服务中心单月累计接听物业、供暖等服务热线1600余个，受理百姓诉求工单830余件，解决率98%，满意率95%。4家供暖企业进一步完善供暖应急保障体系，集中巡查检修供热一次管线超8.7万米，为确保群众温暖过冬奠定坚实基础。
'''
cutter = Cutter()

def profile():
    times = 100000
    now = time.time()
    for i in range(times):
        jieba.lcut(text, HMM=False)
    print("jieba", time.time() - now)
    now = time.time()
    for i in range(times):
        cutter.cutword(text)
    print("cutword", time.time() - now)

if __name__ == "__main__":
    profile()
    print (cutter.cutword(text))


