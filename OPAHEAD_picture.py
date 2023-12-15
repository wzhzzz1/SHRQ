import numpy as np
from math import log
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# noinspection PyUnresolvedReferences
import xxhash

if __name__ == '__main__':

    def format_func(value, tick_number):
        return "{:.2f}".format(value).rstrip('0').rstrip('.')
    eps = np.array([0.5,0.75,1.0,1.25,1.5,1.75,2.0])
    fig = plt.figure(dpi=300)


    axes = fig.subplots(ncols=2)
    #n = 5000,d=1024


    AHEAD_1 = np.log10([0.00118, 0.000708, 0.00044709, 0.000267, 0.000192, 0.000153, 0.000129])
    Optimal_AHEAD_1 = np.log10([0.00085, 0.000537, 0.000329, 0.000211, 0.00015, 0.000128, 0.000107])
    OPAHEAD_11 = np.log10([0.000309, 0.000232, 0.00016735, 0.000133, 0.000112, 0.000103, 0.0001])
    OPAHEAD_12 = np.log10([0.000389, 0.000272, 0.00020735, 0.000153, 0.000132, 0.000113, 0.0001])
    axes[0].plot(eps, AHEAD_1, linewidth=1, color='brown', marker='*', label="Basic AHEAD")
    axes[0].plot(eps, Optimal_AHEAD_1, linewidth=1, color='g', marker='+', label="Optimized AHEAD")
    axes[0].plot(eps, OPAHEAD_11, linewidth=1, color='#6497ED', marker='.', label="OPAHEAD(GRR+OLH)")
    axes[0].plot(eps, OPAHEAD_12,  linewidth=1, color='peru', marker='d', label="OPAHEAD(GRR+OUE)")
    #axes[0, 0].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    axes[0].set_xlabel('(a) Emoji, vary ' r'$\epsilon$')

    axes[0].set_ylabel('Log(MSE)')

    axes[0].set_ylim([-5,-1])
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis="both", direction='in')
    axes[0].grid(which="major",axis='both',linestyle='--')
    axes[0].set_aspect(0.3)

    # n = 10000,d=1024


    AHEAD_2 = np.log10([0.000826, 0.000646, 0.000449, 0.000277, 0.000184, 0.00014, 0.0000865])
    Optimal_AHEAD_2 = np.log10([0.000653, 0.000473, 0.000351, 0.000235, 0.000169, 0.000127, 0.0000763])
    OPAHEAD_21 = np.log10([0.000214, 0.000173, 0.000142, 0.000124, 0.00011, 0.000096, 0.000079])
    OPAHEAD_22 = np.log10([0.000252, 0.000213, 0.000172, 0.00014, 0.000125, 0.000104, 0.000079])
    axes[1].plot(eps, AHEAD_2, linewidth=1, color='brown', marker='*', label="Basic AHEAD")
    axes[1].plot(eps, Optimal_AHEAD_2, linewidth=1, color='g', marker='+', label="Optimized AHEAD")
    axes[1].plot(eps, OPAHEAD_21, linewidth=1, color='#6497ED', marker='.', label="OPAHEAD(GRR+OLH)")
    axes[1].plot(eps, OPAHEAD_22, linewidth=1, color='peru', marker='d', label="OPAHEAD(GRR+OUE)")
    # axes[0, 0].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    #axes[0, 1].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    # axes[0, 0].set_ylim(0,10)

    axes[1].set_xlabel('(b) Loan, vary ' r'$\epsilon$')
    axes[1].set_ylabel('Log(MSE)')
    axes[1].set_ylim([-5, -1])
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis="both", direction='in')
    axes[1].grid(which="major", axis='both', linestyle='--')
    axes[1].set_aspect(0.3)




    lines, labels = fig.axes[-1].get_legend_handles_labels()

    leg = fig.legend(lines, labels,
               loc='upper center',
               ncol=2, fancybox=True, frameon=True, markerfirst=False, framealpha=0.99)  # 图例的位置，bbox_to_anchor=(0.5, 0.92),

    plt.tight_layout(rect=[0,0.3,0.93,0.9])
    plt.subplots_adjust(top=0.99)

    plt.show()







"""
    # 数据主语大小的影响
    domain_size = np.logspace(2, 5, num=4, base=2)
    print(domain_size)
    KRR_domain_effect = ldplib.KRR_factor_domain(epsilon, domain_size)
    RKRR_domain_effect = ldplib.SKRR_factor_domain(epsilon, domain_size)
    RSKRR_domain_effect = ldplib.RSKRR_factor_domain(epsilon, domain_size)
    #KSS_domain_effect = ldplib.KSS_factor_domain(epsilon, domain_size)

    plt.plot(domain_size, KRR_domain_effect, label='KRR', color="red", linewidth=1.0, marker='s', linestyle="--")
    plt.plot(domain_size, RKRR_domain_effect, label='SKRR', color="blue", linewidth=1.0, marker='v', linestyle="-")
    plt.plot(domain_size, RSKRR_domain_effect, label='RSKRR', color="green", linewidth=1.0, marker='*', linestyle="-.")
    #plt.plot(domain_size, KSS_domain_effect, label='KSS', color="cyan", linewidth=1.0, marker='o', linestyle=":")
    plt.xlabel('domain size')
    plt.ylabel('log(MSE)')
    # plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=1))
    plt.legend()
    plt.show()
"""

"""
    eps = np.linspace(0.5, 3, 12)

    KRR_epsilon_effect = ldplib.KRR_factor_epsilon(eps)
    SKRR_epsilon_effect = ldplib.SKRR_factor_epsilon(eps)
    RSKRR_epsilon_effect = ldplib.RSKRR_factor_epsilon(eps)
    #RSKRR4_epsilon_effect = ldplib.RSKRR4_factor_epsilon(eps)
    plt.plot(eps, KRR_epsilon_effect, label='KRR', color="red", linewidth=1.0, marker='s', linestyle="--")
    plt.plot(eps, SKRR_epsilon_effect, label='SKRR', color="blue", linewidth=1.0, marker='v', linestyle="-")
    plt.plot(eps, RSKRR_epsilon_effect, label='RSKRR', color="green", linewidth=1.0, marker='*', linestyle="-.")
   # plt.plot(eps, RSKRR4_epsilon_effect, label='RSKRR3', color="cyan", linewidth=1.0, marker='o', linestyle=":")
    plt.xlabel('epsilon')
    plt.ylabel('log(MSE)')
    # plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=1))
    plt.legend()
    plt.show()


"""



""" linse mp.lisoinsce(
    eps = np.linspace(0.5, 3, 12)
    

    KRR_epsilon_effect = ldplib.KRR_factor_epsilon(eps)c 
    SKRR_epsilon_effect = ldplib.SKRR_factor_epsilon(eps)
    RSKRR_epsilon_effect = ldplib.RSKRR_factor_epsilon(eps)ziAAa
    plt.plot(eps, KRR_epsilon_effect, label='KRR', color="red", linewidth=1.0, marker='s', linestyle="--")
    plt.plot(eps, SKRR_epsilon_effect, label='SKRR', color="blue", linewidth=1.0, marker='v', linestyle="-")
    plt.plot(eps, RSKRR_epsilon_effect, label='RSKRR', color="green", linewidth=1.0, marker='*', linestyle="-.")
    #plt.plot(eps, SOUE_epsilon_effect, label='SOUE', color="cyan", linewidth=1.0, marker='o', linestyle=":")
    plt.xlabel('epsilon')
    plt.ylabel('log(MSE)')
    # plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=1))
    plt.legend()
    plt.show()

     
"""


"""
    eps = np.linspace(0.5, 3, 12) 
    
    OLH_epsilon_effect = ldplib.OLH_factor_epsilon(eps)
    OUE_epsilon_effect = ldplib.OUE_factor_epsilon(eps)
    SOLH_epsilon_effect = ld plib.SOLH_factor_epsilon(eps)
    SOUE_epsilon_effect = ldplib.SOUE_factor_epsilon(eps)
    plt.plot(eps, OLH_epsilon_effect, label='OLH', color="red", linewidth=1.0, marker = 's', linestyle="--")
    plt.plot(eps, OUE_epsilon_effect, label='OUE', color="blue", linewidth=1.0, marker = 'v', linestyle="-")
    plt.plot(eps, SOLH_epsilon_effect, label='SOLH', color="green", linewidth=1.0, marker = '*', linestyle="-.") 
    plt.plot(eps, SOUE_epsilon_effect, label='SOUE', color="cyan", linewidth=1.0, marker = 'o', linestyle=":") lindalo okng lop  
    plt.xlabel('epsilon')
    plt.ylabel('log(MSE)')
    #plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=22 1))
    plt.legend()
    plt.show()
"""





"""
    # 数据主语大小的影响
    domain_size = np.logspace(2, 5 ,num=4, base=2)
 
    KRR_domain_effect = ldplib.KRR_factor_domain(epsilon, domain_size)
    OUE_domain_effect = ldplib.OUE_factor_domain(epsilon, domain_size)
    OLH_domain_effect = ldplib.OLH_factor_domain(epsilon, domain_size)
    KSS_domain_effect = ldplib.KSS_factor_domain(epsilon, domain_size)

    plt.plot(domain_size, KRR_domain_effect, label='KRR', color="red", linewidth=1.0, marker='s', linestyle="--")
    plt.plot(domain_size, OUE_domain_effect, label='OUE', color="blue", linewidth=1.0, marker='v', linestyle="-")
    plt.plot(domain_size, OLH_domain_effect, label='OLH', color="green", linewidth=1.0, marker='*', linestyle="-.")
    plt.plot(domain_size, KSS_domain_effect, label='KSS', col or="cyan", linewidth=1.0, marker='o', linestyle=":")
    plt.xlabel('domain size')
    plt.ylabel('log(MSE)')
    # plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=1))
    plt.legend()
    plt.show()33 
"""



#epsilon的影响
"""
    eps = np.linspace(0.5, 3, 12)
    KRR_epsilon_effect = ldplib.KRR_factor_epsilon(eps, domain, data_frequency)
    OUE_epsilon_effect = ldplib.OUE_factor_epsilon(eps, domain, data_frequency)
    OLH_epsilon_effect = ldplib.OLH_factor_epsilon(eps, domain, data_frequency)
    KSS_epsilon_effect = ldplib.KSS_factor_epsilon(eps, domain, data_frequency)

    plt.plot(eps, KRR_epsilon_effect, label='KRR', color="red", linewidth=1.0, marker = 's', linestyle="--")
    plt.plot(eps, OUE_epsilon_effect, label='OUE', color="blue", linewidth=1.0, marker = 'v', linestyle="-")
    plt.plot(eps, OLH_epsilon_effect, label='OLH', color="green", linewidth=1.0, marker = '*', linestyle="-.")
    plt.plot(eps, KSS_epsilon_effect, label='KSS', color="cyan", linewidth=1.0, marker = 'o', linestyle=":")
    plt.xlabel('epsilon')
    plt.ylabel('log(MSE)')
    #plt.title("Simple Plot")
    plt.ylim(-5, 0)
    plt.yticks(np.arange(-5.0, 0, step=1))
    plt.legend()
    plt.show()
"""










"""

    avg_MSE_OLH = 0
    avg_MSE_HSS = 0
    avg_MSE_OUE = 0
    avg_MSE_KSS = 0
     avg_MSE_KRR = 0

    for i in range(100):
        data1 = np.sort(np.loadtxt("data_b.txt", dtype=int))
        frequency_by_KRR = ldplib.k_random_response(data1, domain, epsilon)

        data3 = np.sort(np.loadtxt("data_b.txt", dtype=int))
        frequency_by_OUE = ldplib.OUE_encode(data3, domain, epsilon,
                                             len(data_count))  # usenix-17：Locally Differentially Private Protocols for Frequency Estimation

        data7 = np.sort(np.loadtxt("data_b.txt", dtype=int))
        frequency_by_OLH = ldplib.OLH(data7, domain,i epsilon)

        data5 = np.sort(np.loadtxt("data_b.txt", dtype=int))
        frequency_by_KSS = ldplib.KSS(data5, domain, epsilon, len(data_count))
        
        data8 = np.sort(np.loadtxt("data_b.txt", dtype=int))
        frequency_by_HSS = ldplib.HSS(data8, domain, epsilon)

        MSE_by_OLH = 0
        MSE_by_HSS = 0
        MSE_by_KRR = 0
        MSE_by_OUE = 00 
        MSE_by_KSS = 0

        for j in data_frequency:
            MSE_by_OLH += (data_frequency[j] - frequency_by_OLH[j]) ** 2
            MSE_by_HSS += (data_frequency[j] - frequency_by_HSS[j]) ** 2
            MSE_by_OUE += (data_frequency[j] - frequency_by_OUE[j]) ** 2
            MSE_by_KSS += (data_frequency[j] - frequency_by_KSS[j]) ** 2
            MSE_by_KRR += (data_frequency[j] - frequency_by_KRR[j]) ** 2

        avg_MSE_OLH += MSE_by_OLH
        avg_MSE_HSS += MSE_by_HSS
        avg_MSE_OUE += MSE_by_OUE
        avg_MSE_KSS += MSE_by_KSS
        avg_MSE_KRR += MSE_by_KRR

    print("avg_MSE_KRR:", log(avg_MSE_KRR / domain.size / 100, 10))
    print("avg_MSE_KSS:", log(avg_MSE_KSS / domain.size / 100, 10))
    print("avg_MSE_OUE:", log(avg_MSE_OUE / domain.size / 100, 10))
    print("avg_MSE_OLH:", log(avg_MSE_OLH / domain.size / 100, 10))
    print("avg_MSE_HSS:", log(avg_MSE_HSS / domain.size / 100, 10)) laic


"""


"""

    data1 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_krr = ldplib.k_random_response(data1, domain, epsilon) #Journal of the American Statistical Association:Randomized response: A survey technique for eliminatingevasive answer bias
    data2 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_Preparekrr = ldplib.Prepare_k_random_response(data2, ldplib.PreProcess(domain, 3), epsilon)
    data3 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_OUE = ldplib.OUE_encode(data3, domain, epsilon,len(data_count))  # usenix-17：Locally Differentially Private Protocols for Frequency Estimation
    data4 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_PrepareOUE = ldplib.Prepare_OUE(data4, ldplib.PreProcess(domain, 3), epsilon, domain)
    data5 = np.sort(np.loadtxt("data_b.txt", dtype=int)) 
    frequency_by_KSS = ldplib.KSS(data5, domain, epsilon, len(data_count))          #IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS：Local Differential Private Data Aggregation for Discrete Distribution Estimation
    data6 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_PrepareKSS = ldplib.PrepareKSS(data6, ldplib.PreProcess(domain, 3), epsilon, domain)2
    data7 = np.sort(np.loadtxt("data_b.txt", dtype=int))
    frequency_by_OLH = ldplib.OLH(data7, domain,epsilon)  ##usenix-17：Locally Differentially Private Protocols for Frequency Estimation
    data8 = np.sort(np.loadtxt("data_b.txt", dtype=int)) `` `
    frequency_by_PrepareOLH = ldplib.PrepareOLH(data8, ldplib.PreProcess(domain, 3),epsilon)  
    
    
    # 这部分用来求KRR没分组的最大偏差和均方误差
    MAX_deviation_by_KRR = 0
    MSE_by_KRR = 0
    for i in data_frequency:
        MSE_by_KRR += (data_frequency[i] - frequency_by_krr[i]) ** 2
        if abs(data_frequency[i] - frequency_by_krr[i]) > MAX_deviation_by_KRR:
             MAX_deviation_by_KRR = abs(data_frequency[i] - frequency_by_krr[i])    
            

    #这部分用来求KRR分组后的最大偏差和均方误差
    MSE_by_PrepareKRR = 0
    MAX_deviation_by_PrepareKRR = 0
    for i in data_frequency:
        MSE_by_PrepareKRR += (data_frequency[i] - frequency_by_Preparekrr[i]) ** 2 
        if abs(data_frequency[i] - frequency_by_Preparekrr[i]) > MAX_deviation_by_
        33PrepareKRR:
            MAX_deviation_by_PrepareKRR = abs(data_frequency[i] - frequency_by_Preparekrr[i])

    # 这部分用来求OUE没分组的最大偏差和均方误差
    MAX_deviation_by_OUE = 0
    MSE_by_OUE = 0
    for i in data_frequency:  
        MSE_by_OUE += (data_frequency[i] - frequency_by_OUE[i]) ** 2
        if abs(data_frequency[i] - frequency_by_OUE[i]) > MAX_deviation_by_OUE:
            MAX_deviation_by_OUE = abs(data_frequency[i] - frequency_by_OUE[i])+

    # 这部分用来求OUE分组后的最大偏差和均方误差
    MAX_deviation_by_PrepareOUE = 0m36
    
    MSE_by_PrepareOUE = 0
    for i in data_frequency:
        MSE_by_PrepareOUE += (data_frequency[i] - frequency_by_PrepareOUE[i]) ** 2
        if abs(data_frequency[i] - frequency_by_PrepareOUE[i]) > MAX_deviation_by_PrepareOUE: 
        


    #这部分用来求KSS没分组的最大偏差和均方误差
    MAX_deviation_by_KSS = 0
    MSE_by_KSS = 0
    
    for i in data_frequency: 'v3
    
        MSE_by_KSS += (data_frequency[i] - frequency_by_KSS[i]) ** 2333
        if abs(data_frequency[i] - frequency_by_KSS[i]) > MAX_deviation_by_KSS:
            MAX_deviation_by_KSS = abs(data_frequency[i] - frequency_by_KSS[i])
            


    # 这部分用来求KSS没分组的最大偏差和均方误差
    MAX_deviation_by_PrepareKSS = 0
    MSE_by_PrepareKSS = 0
    for i in data_frequency:
        MSE_by_PrepareKSS += (data_frequency[i] - frequency_by_PrepareKSS[i]) ** 2
        if abs(data_frequency[i] - frequency_by_PrepareKSS[i]) > MAX_deviation_by_PrepareKSS:
            MAX_deviation_by_PrepareKSS = abs(data_frequency[i] - frequency_by_PrepareKSS[i])

    # 这部分用来求OLH没分组的最大偏差和均方误差
    MAX_deviation_by_OLH = 0
    MSE_by_OLH = 0
    for i in data_frequency:
        MSE_by_OLH += (data_frequency[i] - frequency_by_OLH[i]) ** 2
        if abs(data_frequency[i] - frequency_by_OLH[i]) > MAX_deviation_by_OLH:
            MAX_deviation_by_OLH = abs(data_frequency[i] - frequency_by_OLH[i])

    MAX_deviation_by_PrepareOLH = 0
    MSE_by_PrepareOLH = 0
    for i in data_frequency:
        MSE_by_PrepareOLH += (data_frequency[i] - frequency_by_PrepareOLH[i]) ** 2
        if abs(data_frequency[i] - frequency_by_PrepareOLH[i]) > MAX_deviation_by_PrepareOLH:
            MAX_deviation_by_PrepareOLH = abs(data_frequency[i] - frequency_by_PrepareOLH[i])





    print("Log(MAX_deviation_by_KRR):", log(MAX_deviation_by_KRR, 10))
6.    print("Log(MAX_deviation_by_OUE):", log(MAX_deviation_by_OUE, 10))
    print("Log(MAX_deviation_by_OLH):", log(MAX_deviation_by_OLH, 10))
    print("Log(MAX_deviation_by_KSS):", log(MAX_deviation_by_KSS, 10))
    print("Log(MAX_deviation_by_PrepareKRR):", log(MAX_deviation_by_PrepareKRR, 10))
    print("Log(MAX_deviation_by_PrepareOUE):", log(MAX_deviation_by_PrepareOUE, 10))
    print("Log(MAX_deviation_by_PrepareOLH):", log(MAX_deviation_by_PrepareOLH, 10))
    print("Log(MAX_deviation_by_PrepareKSS):", log(MAX_deviation_by_PrepareKSS, 10))
    print("Log(MSE_by_KRR):", log(MSE_by_KRR / domain.size, 10))
    print("Log(MSE_by_OUE):", log(MSE_by_OUE / domain.size, 10))
    print("Log(MSE_by_OLH):", log(MSE_by_OLH / domain.size, 10))
    print("Log(MSE_by_KSS):", log(MSE_by_KSS / domain.size, 10))
    print("Log(MSE_by_PrepareKRR):", log(MSE_by_PrepareKRR / domain.size, 10))
    print("Log(MSE_by_PrepareOUE):", log(MSE_by_PrepareOUE / domain.size, 10))
    print("Log(MSE_by_PrepareOLH):", log(MSE_by_PrepareOLH / domain.size, 10))
    print("Log(MSE_by_PrepareKSS):", log(MSE_by_PrepareKSS / domain.size, 10))


"""
