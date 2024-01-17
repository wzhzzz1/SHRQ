import random

# 生成内容并写入文件
num = 0
with open('query_table/rand_large_query_domain7_attribute1.txt', 'w') as file:
    while(num<200):
        a = random.randint(0, 1023)
        b = random.randint(a + 1, 1024)  # 确保b大于a
        if (b-a)/1024>0.5:
            file.write(f"{a} {b}\n")
            num+=1
