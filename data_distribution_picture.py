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


    axes = fig.subplots(nrows=2, ncols=2)

    HIO_1 = np.log10([0.00305, 0.00134, 0.00079, 0.000463, 0.000284, 0.0002196, 0.000146])
    AHEAD_1 = np.log10([0.00098, 0.000708, 0.00044709, 0.000267, 0.000192, 0.000153, 0.000129])
    Optimal_AHEAD_1 = np.log10([0.00085, 0.000537, 0.000329, 0.000211, 0.00015, 0.000128, 0.000107])
    OPAHEAD_1 = np.log10([0.000412, 0.000276, 0.000213, 0.000171, 0.000139, 0.000124, 0.000105])
    axes[0, 0].plot(eps, HIO_1, linewidth=1, color='brown', marker='*', label="HIO")
    axes[0, 0].plot(eps, AHEAD_1, linewidth=1, color='g', marker='+', label="AHEAD")
    axes[0, 0].plot(eps, Optimal_AHEAD_1, linewidth=1, color='#6497ED', marker='.', label="Optimized AHEAD")
    axes[0, 0].plot(eps, OPAHEAD_1, linewidth=1, color='peru', marker='d', label="OPAHEAD")
    #axes[0, 0].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    #axes[0, 0].set_xlabel('(a) n=5000, vary ' r'$\epsilon$')
    axes[0, 0].set_xlabel('(a) Normal distribution, vary ' r'$\epsilon$')
    axes[0, 0].set_ylabel('Log(MSE)')

    axes[0, 0].set_ylim([-5,-1])
    axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[0, 0].xaxis.set_major_formatter(formatter)
    axes[0, 0].tick_params(axis="both", direction='in')
    axes[0,0].grid(which="major",axis='both',linestyle='--')
    axes[0,0].set_aspect(0.3)

    # n = 10000,d=1024
    #HIO_2 = np.log10([0.0305,0.0141,0.00741,0.00466,0.002887,0.00216,0.00133])
    #Optimal_HIO_2 = np.log10([0.0291,0.0125,0.00729,0.00415,0.00263,0.00183,0.00112])
    #AHEAD_2 = np.log10([0.0042,0.00218,0.0017,0.00124,0.00115,0.000978,0.00086])
    #Optimal_AHEAD_2 = np.log10([0.00346,0.00175,0.001358,0.001038,0.000945,0.000784,0.00057])
    HIO_2 = np.log10([0.00305, 0.00134, 0.00079, 0.000463, 0.000284, 0.0002196, 0.000146])
    AHEAD_2 = np.log10([0.00118, 0.000708, 0.00044709, 0.000267, 0.000192, 0.000153, 0.000129])
    Optimal_AHEAD_2 = np.log10([0.00085, 0.000537, 0.000329, 0.000211, 0.00015, 0.000128, 0.000107])
    OPAHEAD_2 = np.log10([0.000343, 0.000264, 0.0002115, 0.000174, 0.000140, 0.0001226, 0.000110])
    axes[0, 1].plot(eps, HIO_2, linewidth=1, color='brown', marker='*', label="HIO")
    axes[0, 1].plot(eps, AHEAD_2, linewidth=1, color='g', marker='+', label="AHEAD")
    axes[0, 1].plot(eps, Optimal_AHEAD_2, linewidth=1, color='#6497ED', marker='.', label="Optimized AHEAD")
    axes[0, 1].plot(eps, OPAHEAD_2, linewidth=1, color='peru', marker='d', label="OPAHEAD")
    # axes[0, 0].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    #axes[0, 1].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    # axes[0, 0].set_ylim(0,10)
    axes[0, 1].set_xlabel('(b) Skewed distribution, vary ' r'$\epsilon$')
    #axes[0, 1].set_xlabel('(b) n=10000, vary ' r'$\epsilon$')
    axes[0, 1].set_ylabel('Log(MSE)')
    axes[0, 1].set_ylim([-5, -1])
    axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[0, 1].xaxis.set_major_formatter(formatter)
    axes[0, 1].tick_params(axis="both", direction='in')
    axes[0, 1].grid(which="major", axis='both', linestyle='--')
    axes[0, 1].set_aspect(0.3)

    # n = 50000,d=1024
    #HIO_3 = np.log10([0.006029,0.0026,0.0014,0.00085,0.0005859,0.0004,0.000278])
    #Optimal_HIO_3 = np.log10([0.00581,0.00248,0.001316,0.00073,0.000532,0.0003655,0.000265])
    #AHEAD_3 = np.log10([0.00147,0.000834,0.000605,0.00042,0.000319,0.000277,0.000214])
    #Optimal_AHEAD_3 = np.log10([0.00122,0.00064,0.0004736,0.000313,0.0002714,0.000216,0.000174])
    HIO_3 = np.log10([0.00315, 0.00144, 0.00073, 0.000463, 0.000284, 0.0002196, 0.000146])
    AHEAD_3 = np.log10([0.00098, 0.000708, 0.00044709, 0.000267, 0.000192, 0.000153, 0.000129])
    Optimal_AHEAD_3 = np.log10([0.00075, 0.000507, 0.000329, 0.000211, 0.00015, 0.000128, 0.000107])
    OPAHEAD_3 = np.log10([0.000353, 0.000264, 0.000208, 0.0001732, 0.0001390, 0.00012, 0.000096])
    axes[1, 0].plot(eps, HIO_3, linewidth=1, color='brown', marker='*', label="HIO")
    axes[1, 0].plot(eps, AHEAD_3, linewidth=1, color='g', marker='+', label="AHEAD")
    axes[1, 0].plot(eps, Optimal_AHEAD_3, linewidth=1, color='#6497ED', marker='.', label="Optimized AHEAD")
    axes[1, 0].plot(eps, OPAHEAD_3, linewidth=1, color='peru', marker='d', label="OPAHEAD")
    # axes[0, 0].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    # axes[0, 0].set_ylim(0,10)
    #axes[1, 0].set_xlabel('(c) n=50000, vary ' r'$\epsilon$')
    axes[1, 0].set_xlabel('(c) Exponential distribution, vary ' r'$\epsilon$')
    axes[1, 0].set_ylabel('Log(MSE)')
    axes[1, 0].set_ylim([-5, -1])
    axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[1, 0].xaxis.set_major_formatter(formatter)
    axes[1, 0].tick_params(axis="both", direction='in')
    axes[1, 0].grid(which="major", axis='both', linestyle='--')
    axes[1, 0].set_aspect(0.3)



    HIO_4=np.log10([0.00291,0.00129,0.000725,0.000475,0.0002887,0.000201,0.000169])
    AHEAD_4=np.log10([0.00132,0.00076,0.00048,0.000358	,0.000243,0.000207,0.000142])
    Optimal_AHEAD_4 = np.log10([0.00115,0.00064,0.00039,0.000315,0.00023,0.000187,0.000139])
    OPAHEAD_4 = np.log10([0.000618, 0.00040, 0.000307, 0.000254, 0.000202, 0.000173, 0.000132])
    axes[1, 1].plot(eps, HIO_4, linewidth=1, color='brown', marker='*', label="HIO")
    axes[1, 1].plot(eps, AHEAD_4, linewidth=1, color='g', marker='+', label="AHEAD")
    axes[1, 1].plot(eps, Optimal_AHEAD_4, linewidth=1, color='#6497ED', marker='.', label="Optimized AHEAD")
    axes[1, 1].plot(eps, OPAHEAD_4, linewidth=1, color='peru', marker='d', label="OPAHEAD")
    #axes[1, 1].plot(x, x ** 3, linewidth=1, color='b', marker='+', label="KSS")
    # axes[0, 0].set_ylim(0,10)
    #axes[1, 1].set_xlabel('(d) n=100000, vary ' r'$\epsilon$' )
    axes[1, 1].set_xlabel('(d) Uniform distribution, vary ' r'$\epsilon$')
    axes[1, 1].set_ylabel('Log(MSE)')
    axes[1, 1].set_ylim([-5, -1])
    axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(7))
    formatter = ticker.FuncFormatter(format_func)
    axes[1, 1].xaxis.set_major_formatter(formatter)
    axes[1 ,1].tick_params(axis="both", direction='in')
    axes[1, 1].grid(which="major", axis='both', linestyle='--')
    axes[1, 1].set_aspect(0.3)


    lines, labels = fig.axes[-1].get_legend_handles_labels()

    leg = fig.legend(lines, labels,
               loc='upper center',
               ncol=4, fancybox=True, frameon=True, markerfirst=False, framealpha=0.99)  # 图例的位置，bbox_to_anchor=(0.5, 0.92),

    plt.tight_layout(rect=[0,0.05,0.93,0.9])
    plt.subplots_adjust(top=0.88)

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
