import pickle
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

'''population = pickle.load(open('./Results2_vgg19_rand_fc/population_train.pkl', 'rb'))
count = 0
pivot = 0
pivot1 = 0
pivot2 = 0

#function2代表模型正确率
res1 = []
#function1代表模型剪枝率
res2 = []
max_acc = 0
pru = 0
min_acc = 100
min_pru = 0
max_pru = 0
acc= 0
for ind in population:
    count += 1
    if max_acc < 100 - ind.obj[0]:
        max_acc = 100 - ind.obj[0]
        pivot = count - 1
        pru = 1 - ind.obj[1] / 13706
    
    if 100 - ind.obj[0] < min_acc:
        min_acc = 100 - ind.obj[0]
        pivot1 = count - 1
        min_pru = 1 - ind.obj[1] / 13706
        
    if max_pru < 1 - ind.obj[1] / 13706:
        max_pru = 1 - ind.obj[1] / 13706
        acc = 100 - ind.obj[0]
        pivot2 = count - 1
        
    res1.append(ind.obj[0])
    res2.append(1 - ind.obj[1] / 13706)
    
#print('f1:', res1)
#print('f2', res2)
plt.xlabel('Pruning rate (#)', fontsize=10)
plt.ylabel('Error rate (%)', fontsize=10)
p1 = plt.scatter(res2, res1, s = 5, c = 'y', marker = 's')


#legend = plt.legend([p1], ["solution (300 epochs)"], loc = 'best')
#plt.show()  


population = pickle.load(open('./Results2_vgg19_rand_fc/population.pkl', 'rb'))
count = 0
pivot = 0
pivot1 = 0
pivot2 = 0

#function2代表模型正确率
res1 = []
#function1代表模型剪枝率
res2 = []
max_acc = 0
pru = 0
min_acc = 100
min_pru = 0
max_pru = 0
acc= 0
for ind in population:
    count += 1
    if max_acc < 100 - ind.obj[0]:
        max_acc = 100 - ind.obj[0]
        pivot = count - 1
        pru = 1 - ind.obj[1] / 13706
    
    if 100 - ind.obj[0] < min_acc:
        min_acc = 100 - ind.obj[0]
        pivot1 = count - 1
        min_pru = 1 - ind.obj[1] / 13706
        
    if max_pru < 1 - ind.obj[1] / 13706:
        max_pru = 1 - ind.obj[1] / 13706
        acc = 100 - ind.obj[0]
        pivot2 = count - 1
        
    res1.append(ind.obj[0])
    res2.append(1 - ind.obj[1] / 13706)

#print('f1:', res1)
#print('f2', res2)
plt.xlabel('Pruning rate (#)', fontsize=10)
plt.ylabel('Error rate (%)', fontsize=10)
p2 = plt.scatter(res2, res1, s = 5, c = 'r', marker = '^')
#legend = plt.legend([p2], ["solution (60 epochs)"], loc = 'best')
'''
population = pickle.load(open('./Results_vgg19_rand_fc/population-500.pkl', 'rb'))
count = 0
pivot = 0
pivot1 = 0
pivot2 = 0

#function2代表模型正确率
res1 = []
#function1代表模型剪枝率
res2 = []
max_acc = 0
pru = 0
min_acc = 100
min_pru = 0
max_pru = 0
acc= 0
for ind in population:
    count += 1
    if max_acc < 100 - ind.obj[0]:
        max_acc = 100 - ind.obj[0]
        pivot = count - 1
        pru = 1 - ind.obj[1] / 13706
    
    if 100 - ind.obj[0] < min_acc:
        min_acc = 100 - ind.obj[0]
        pivot1 = count - 1
        min_pru = 1 - ind.obj[1] / 13706
        
    if max_pru < 1 - ind.obj[1] / 13706:
        max_pru = 1 - ind.obj[1] / 13706
        acc = 100 - ind.obj[0]
        pivot2 = count - 1
        
    res1.append(ind.obj[0])
    res2.append(1 - ind.obj[1] / 13706)

#print('f1:', res1)
#print('f2', res2)
plt.xlabel('Pruning rate (#)', fontsize=10)
plt.ylabel('Error rate (%)', fontsize=10)
p3 = plt.scatter(res2, res1, s = 5, c = 'b', marker = 'o')
legend = plt.legend([p3], ["solution (Gen 500)"], loc = 'best')
#legend = plt.legend([p1, p2, p3], ["solution (epochs 300)", "solution (epochs 60)", "solution (Gen 500)"], loc = 'best' )
plt.show()        









